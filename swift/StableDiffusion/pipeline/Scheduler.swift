// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Accelerate
import CoreML

public protocol Scheduler {
    /// Number of diffusion steps performed during training
    var trainStepCount: Int { get }

    /// Number of inference steps to be performed
    var inferenceStepCount: Int { get }

    /// Training diffusion time steps index by inference time step
    var timeSteps: [Int] { get }

    /// Training diffusion time steps index by inference time step
    func calculateTimesteps(strength: Float?) -> [Int]

    /// Schedule of betas which controls the amount of noise added at each timestep
    var betas: MLTensor { get }

    /// 1 - betas
    var alphas: MLTensor { get }

    /// Cached cumulative product of alphas
    var alphasCumProd: MLTensor { get }

    /// Standard deviation of the initial noise distribution
    var initNoiseSigma: Float { get }

    /// Denoised latents
    var modelOutputs: [MLTensor] { get }

    /// Compute a de-noised image sample and step scheduler state
    ///
    /// - Parameters:
    ///   - output: The predicted residual noise output of learned diffusion model
    ///   - timeStep: The current time step in the diffusion chain
    ///   - sample: The current input sample to the diffusion model
    /// - Returns: Predicted de-noised sample at the previous time step
    /// - Postcondition: The scheduler state is updated.
    ///   The state holds the current sample and history of model output noise residuals
    func step(output: MLTensor, timeStep t: Int, sample s: MLTensor) -> MLTensor
}

public extension Scheduler {
    var initNoiseSigma: Float { 1 }
}

public extension Scheduler {
    func addNoise(
        originalSample: MLTensor,
        noise: [MLTensor],
        strength: Float
    ) -> [MLTensor] {
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        let alphaProdt = alphasCumProd[timeSteps[startStep]]
        let betaProdt = 1 - alphaProdt
        let sqrtAlphaProdt = alphaProdt.squareRoot()
        let sqrtBetaProdt = betaProdt.squareRoot()

        let noisySamples = noise.map {
            originalSample * sqrtAlphaProdt + $0 * sqrtBetaProdt
        }

        return noisySamples
    }
}

// MARK: - Timesteps

public extension Scheduler {
    func calculateTimesteps(strength: Float?) -> [Int] {
        guard let strength else { return timeSteps }
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        let actualTimesteps = Array(timeSteps[startStep...])
        return actualTimesteps
    }
}

// MARK: - BetaSchedule

/// How to map a beta range to a sequence of betas to step over
public enum BetaSchedule {
    /// Linear stepping between start and end
    case linear
    /// Steps using linspace(sqrt(start),sqrt(end))^2
    case scaledLinear
}

// MARK: - PNDMScheduler

/// A scheduler used to compute a de-noised image
///
///  This implementation matches:
///  [Hugging Face Diffusers PNDMScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py)
///
/// This scheduler uses the pseudo linear multi-step (PLMS) method only, skipping pseudo Runge-Kutta (PRK) steps
public final class PNDMScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: MLTensor
    public let alphas: MLTensor
    public let alphasCumProd: MLTensor
    public let timeSteps: [Int]

    public let alpha_t: MLTensor
    public let sigma_t: MLTensor
    public let lambda_t: MLTensor

    public private(set) var modelOutputs: [MLTensor] = []

    // Internal state
    var counter: Int
    var ets: [MLTensor]
    var currentSample: MLTensor?

    /// Create a scheduler that uses a pseudo linear multi-step (PLMS)  method
    ///
    /// - Parameters:
    ///   - stepCount: Number of inference steps to schedule
    ///   - trainStepCount: Number of training diffusion steps
    ///   - betaSchedule: Method to schedule betas from betaStart to betaEnd
    ///   - betaStart: The starting value of beta for inference
    ///   - betaEnd: The end value for beta for inference
    /// - Returns: A scheduler ready for its first step
    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount

        switch betaSchedule {
        case .linear:
            self.betas = MLTensor(linearSpaceFrom: betaStart, through: betaEnd, count: trainStepCount)
        case .scaledLinear:
            self.betas = MLTensor(
                linearSpaceFrom: betaStart.squareRoot(),
                through: betaEnd.squareRoot(),
                count: trainStepCount
            ).squared()
        }
        self.alphas = 1.0 - betas
        self.alphasCumProd = self.alphas.cumulativeProduct()
        let stepsOffset = 1 // For stable diffusion
        let stepRatio = Float(trainStepCount / stepCount )
        let forwardSteps = (0..<stepCount).map {
            Int((Float($0) * stepRatio).rounded()) + stepsOffset
        }

        self.alpha_t = alphasCumProd.squareRoot()
        self.sigma_t = (1 - alphasCumProd).squareRoot()
        self.lambda_t = alpha_t.log() - sigma_t.log()

        var timeSteps: [Int] = []
        timeSteps.append(contentsOf: forwardSteps.dropLast(1))
        timeSteps.append(timeSteps.last!)
        timeSteps.append(forwardSteps.last!)
        timeSteps.reverse()

        self.timeSteps = timeSteps
        self.counter = 0
        self.ets = []
        self.currentSample = nil
    }

    /// Compute a de-noised image sample and step scheduler state
    ///
    /// - Parameters:
    ///   - output: The predicted residual noise output of learned diffusion model
    ///   - timeStep: The current time step in the diffusion chain
    ///   - sample: The current input sample to the diffusion model
    /// - Returns: Predicted de-noised sample at the previous time step
    /// - Postcondition: The scheduler state is updated.
    ///   The state holds the current sample and history of model output noise residuals
    public func step(output: MLTensor, timeStep t: Int, sample s: MLTensor) -> MLTensor {
        var timeStep = t
        let stepInc = (trainStepCount / inferenceStepCount)
        var prevStep = timeStep - stepInc
        var modelOutput = output
        var sample = s

        if counter != 1 {
            if ets.count > 3 {
                ets = Array(ets[(ets.count - 3)..<ets.count])
            }
            ets.append(output)
        } else {
            prevStep = timeStep
            timeStep = timeStep + stepInc
        }

        if ets.count == 1 && counter == 0 {
            modelOutput = output
            currentSample = sample
        } else if ets.count == 1 && counter == 1 {
            modelOutput = (output + ets[back: 1]) / 2.0
            sample = currentSample!
            currentSample = nil
        } else if ets.count == 2 {
            modelOutput = (3 * ets[back: 1] - ets[back: 2]) / 2.0
        } else if ets.count == 3 {
            modelOutput = (23.0 * ets[back: 1] - 16.0 * ets[back: 2] + 5.0 * ets[back: 3]) / 12.0
        } else {
            modelOutput = (55.0 * ets[back: 1] - 59.0 * ets[back: 2] + 37.0 * ets[back: 3] - 9.0 * ets[back: 4]) / 24.0
        }

        let convertedOutput = convertModelOutput(modelOutput: modelOutput, timestep: timeStep, sample: sample)
        modelOutputs.append(convertedOutput)

        let prevSample = previousSample(sample, timeStep, prevStep, modelOutput)

        counter += 1
        return prevSample
    }

    /// Convert the model output to the corresponding type the algorithm needs.
    func convertModelOutput(
        modelOutput: MLTensor,
        timestep: Int,
        sample: MLTensor
    ) -> MLTensor {
        assert(modelOutput.scalarCount == sample.scalarCount)
        let (alpha_t, sigma_t) = (alpha_t[timestep], sigma_t[timestep])
        return (sample - modelOutput * sigma_t) / alpha_t
    }

    /// Compute  sample (denoised image) at previous step given a current time step
    ///
    /// - Parameters:
    ///   - sample: The current input to the model x_t
    ///   - timeStep: The current time step t
    ///   - prevStep: The previous time step t−δ
    ///   - modelOutput: Predicted noise residual the current time step e_θ(x_t, t)
    /// - Returns: Computes previous sample x_(t−δ)
    func previousSample(
        _ sample: MLTensor,
        _ timeStep: Int,
        _ prevStep: Int,
        _ modelOutput: MLTensor
    ) ->  MLTensor {

        // Compute x_(t−δ) using formula (9) from
        // "Pseudo Numerical Methods for Diffusion Models on Manifolds",
        // Luping Liu, Yi Ren, Zhijie Lin & Zhou Zhao.
        // ICLR 2022
        //
        // Notation:
        //
        // alphaProdt       α_t
        // alphaProdtPrev   α_(t−δ)
        // betaProdt        (1 - α_t)
        // betaProdtPrev    (1 - α_(t−δ))
        let alphaProdt = alphasCumProd[timeStep]
        let alphaProdtPrev = alphasCumProd[max(0,prevStep)]
        let betaProdt = 1 - alphaProdt
        let betaProdtPrev = 1 - alphaProdtPrev

        // sampleCoeff = (α_(t−δ) - α_t) divided by
        // denominator of x_t in formula (9) and plus 1
        // Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        // sqrt(α_(t−δ)) / sqrt(α_t))
        let sampleCoeff = (alphaProdtPrev / alphaProdt).squareRoot()

        // Denominator of e_θ(x_t, t) in formula (9)
        let modelOutputDenomCoeff = alphaProdt * betaProdtPrev.squareRoot()
            + (alphaProdt * betaProdt * alphaProdtPrev).squareRoot()

        // full formula (9)
        let modelCoeff = -(alphaProdtPrev - alphaProdt) / modelOutputDenomCoeff
        let prevSample = sample * sampleCoeff + modelOutput * modelCoeff

        return prevSample
    }
}

extension Collection {
    /// Collection element index from the back. *self[back: 1]* yields the last element
    public subscript(back i: Int) -> Element {
        return self[index(endIndex, offsetBy: -i)]
    }
}
