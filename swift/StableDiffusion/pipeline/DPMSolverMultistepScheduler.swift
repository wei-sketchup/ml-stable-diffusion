// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. and The HuggingFace Team. All Rights Reserved.

import Accelerate
import CoreML

/// How to space timesteps for inference
public enum TimeStepSpacing {
    case linspace
    case leading
    case karras
}

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers DPMSolverMultistepScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py)
///
/// It uses the DPM-Solver++ algorithm: [code](https://github.com/LuChengTHU/dpm-solver) [paper](https://arxiv.org/abs/2211.01095).
/// Limitations:
///  - Only implemented for DPM-Solver++ algorithm (not DPM-Solver).
///  - Second order only.
///  - Assumes the model predicts epsilon.
///  - No dynamic thresholding.
///  - `midpoint` solver algorithm.
public final class DPMSolverMultistepScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: MLTensor
    public let alphas: MLTensor
    public let alphasCumProd: MLTensor
    public let timeSteps: [Int]

    public let alpha_t: MLTensor
    public let sigma_t: MLTensor
    public let lambda_t: MLTensor

    public let solverOrder = 2
    private(set) var lowerOrderStepped = 0

    private var usingKarrasSigmas = false

    /// Whether to use lower-order solvers in the final steps. Only valid for less than 15 inference steps.
    /// We empirically find this trick can stabilize the sampling of DPM-Solver, especially with 10 or fewer steps.
    public let useLowerOrderFinal = true

    // Stores solverOrder (2) items
    public private(set) var modelOutputs: [MLTensor] = []

    /// Create a scheduler that uses a second order DPM-Solver++ algorithm.
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    ///   - timeStepSpacing: How to space time steps
    /// - Returns: A scheduler ready for its first step
    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012,
        timeStepSpacing: TimeStepSpacing = .linspace
    ) async {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount

        switch betaSchedule {
        case .linear:
            betas = MLTensor(linearSpaceFrom: betaStart, through: betaEnd, count: trainStepCount)
        case .scaledLinear:
            betas = MLTensor(
                linearSpaceFrom: pow(betaStart, 0.5),
                through: pow(betaEnd, 0.5),
                count: trainStepCount
            ).squared()
        }

        alphas = 1.0 - betas
        alphasCumProd = alphas.cumulativeProduct()

        switch timeStepSpacing {
        case .linspace:
            let start = Float(0), end = Float(trainStepCount-1), stepCount = stepCount+1
            timeSteps = await MLTensor(linearSpaceFrom: start, through: end, count: stepCount)[1...]
                .reversed(alongAxes: 0)
                .round()
                .shapedArray(of: Float.self).scalars.map { Int($0) }
            alpha_t = alphasCumProd.squareRoot()
            sigma_t = (1.0 - alphasCumProd).squareRoot()
        case .leading:
            let lastTimeStep = trainStepCount - 1
            let stepRatio = lastTimeStep / (stepCount + 1)
            // Creates integer timesteps by multiplying by ratio
            timeSteps = (0...stepCount).map { 1 + $0 * stepRatio }.dropFirst().reversed()
            alpha_t = alphasCumProd.squareRoot()
            sigma_t = (1.0 - alphasCumProd).squareRoot()
        case .karras:
            let sigmas = ((1 - alphasCumProd) / alphasCumProd).squareRoot().reversed()
            let logSigmas = sigmas.log()
            // convert_to_karras
            let sigmaMin = sigmas.min()
            let sigmaMax = sigmas.max()
            let rho: Float = 7
            let ramp = MLTensor(linearSpaceFrom: 0.0, through: 1.0, count: stepCount)
            let minInvRho = sigmaMin.pow(1 / rho)
            let maxInvRho = sigmaMax.pow(1 / rho)
            let karrasSigmas = (maxInvRho + ramp * (minInvRho - maxInvRho)).pow(rho)
            let karrasSigmasSplit = karrasSigmas.split(count: karrasSigmas.shape[0])
            let karrasTimeStepsElements = karrasSigmasSplit.map { sigma in
                sigmaToTimestep(sigma: sigma, logSigmas: logSigmas)
            }
            let karrasTimeSteps = MLTensor(concatenating: karrasTimeStepsElements)
                .round().cast(to: Int32.self)
            timeSteps = await karrasTimeSteps.shapedArray(of: Int32.self).scalars.map { timestep in
                Int(timestep)
            }
            alpha_t = 1.0 / (1.0 + karrasSigmas.squared()).squareRoot()
            sigma_t = karrasSigmas * alpha_t
            usingKarrasSigmas = true
        }

        lambda_t = alpha_t.log() - sigma_t.log()
    }

    func timestepToIndex(_ timestep: Int) -> Int {
        guard usingKarrasSigmas else { return timestep }
        return self.timeSteps.firstIndex(of: timestep) ?? 0
    }

    /// Convert the model output to the corresponding type the algorithm needs.
    /// This implementation is for second-order DPM-Solver++ assuming epsilon prediction.
    func convertModelOutput(modelOutput: MLTensor, timestep: Int, sample: MLTensor) -> MLTensor {
        assert(modelOutput.scalarCount == sample.scalarCount)
        let sigmaIndex = timestepToIndex(timestep)
        let (alpha_t, sigma_t) = (alpha_t[sigmaIndex], sigma_t[sigmaIndex])
        return (sample - modelOutput * sigma_t) / alpha_t
    }

    /// One step for the first-order DPM-Solver (equivalent to DDIM).
    /// See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func firstOrderUpdate(
        modelOutput: MLTensor,
        timestep: Int,
        prevTimestep: Int,
        sample: MLTensor
    ) -> MLTensor {
        let prevIndex = timestepToIndex(prevTimestep)
        let currIndex = timestepToIndex(timestep)
        let (p_lambda_t, lambda_s) = (lambda_t[prevIndex], lambda_t[currIndex])
        let p_alpha_t = alpha_t[prevIndex]
        let (p_sigma_t, sigma_s) = (sigma_t[prevIndex], sigma_t[currIndex])
        let h = p_lambda_t - lambda_s
        let x_t = (p_sigma_t / sigma_s) * sample - (p_alpha_t * ((-h).exp() - 1.0)) * modelOutput
        return x_t
    }

    /// One step for the second-order multistep DPM-Solver++ algorithm, using the midpoint method.
    /// var names and code structure mostly follow https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
    func secondOrderUpdate(
        modelOutputs: [MLTensor],
        timesteps: [Int],
        prevTimestep t: Int,
        sample: MLTensor
    ) -> MLTensor {
        let (s0, s1) = (timesteps[back: 1], timesteps[back: 2])
        let (m0, m1) = (modelOutputs[back: 1], modelOutputs[back: 2])
        let (p_lambda_t, lambda_s0, lambda_s1) = (
            lambda_t[timestepToIndex(t)],
            lambda_t[timestepToIndex(s0)],
            lambda_t[timestepToIndex(s1)]
        )
        let p_alpha_t = alpha_t[timestepToIndex(t)]
        let (p_sigma_t, sigma_s0) = (sigma_t[timestepToIndex(t)], sigma_t[timestepToIndex(s0)])
        let (h, h_0) = (p_lambda_t - lambda_s0, lambda_s0 - lambda_s1)
        let r0 = h_0 / h
        let D0 = m0

        let D1 = (1.0 / r0) * (m0 - m1)

        // See https://arxiv.org/abs/2211.01095 for detailed derivations
        let x_t = (
            (p_sigma_t / sigma_s0) * sample
            - (p_alpha_t * ((-h).exp() - 1.0)) * D0
            - 0.5 * (p_alpha_t * ((-h).exp() - 1.0)) * D1
        )
        return x_t
    }

    public func step(output: MLTensor, timeStep t: Int, sample: MLTensor) -> MLTensor {
        let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
        let prevTimestep = stepIndex == timeSteps.count - 1 ? 0 : timeSteps[stepIndex + 1]

        let lowerOrderFinal = useLowerOrderFinal && stepIndex == timeSteps.count - 1 && timeSteps.count < 15
        let lowerOrderSecond = useLowerOrderFinal && stepIndex == timeSteps.count - 2 && timeSteps.count < 15
        let lowerOrder = lowerOrderStepped < 1 || lowerOrderFinal || lowerOrderSecond

        let modelOutput = convertModelOutput(modelOutput: output, timestep: t, sample: sample)
        if modelOutputs.count == solverOrder { modelOutputs.removeFirst() }
        modelOutputs.append(modelOutput)

        let prevSample: MLTensor
        if lowerOrder {
            prevSample = firstOrderUpdate(modelOutput: modelOutput, timestep: t, prevTimestep: prevTimestep, sample: sample)
        } else {
            prevSample = secondOrderUpdate(
                modelOutputs: modelOutputs,
                timesteps: [timeSteps[stepIndex - 1], t],
                prevTimestep: prevTimestep,
                sample: sample
            )
        }
        if lowerOrderStepped < solverOrder {
            lowerOrderStepped += 1
        }

        return prevSample
    }
}

// Implementation based on
// https://github.com/huggingface/diffusers/blob/1f81fbe274e67c843283e69eb8f00bb56f75ffc4/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L444
fileprivate func sigmaToTimestep(sigma: MLTensor, logSigmas: MLTensor) -> MLTensor {
    // get log sigma
    let logSigma = pointwiseMax(sigma, 1e-10).log()
    // get distribution
    let dists = logSigma - logSigmas.expandingShape(at: -1)
    // get sigmas range
    var lowIdx = (dists .>= 0).cumulativeSum(alongAxis: 0).argmax(alongAxis: 0)
    lowIdx = pointwiseMin(lowIdx, MLTensor(Int32(logSigmas.shape[0] - 2)))
    let highIdx = lowIdx + 1

    let low = logSigmas.gathering(atIndices: lowIdx, alongAxis: 0)
    let high = logSigmas.gathering(atIndices: highIdx, alongAxis: 0)

    // interpolate sigmas
    var w = (low - logSigma) / (low - high)
    w = w.clamped(to: 0.0...1.0)
    // transform interpolation to time range
    var t = (1.0 - w) * lowIdx + w * highIdx
    t = t.reshaped(to: sigma.shape)
    return t
}
