#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import logging
import argparse
import shutil
import os
import re
import coremltools as ct
from python_coreml_stable_diffusion import torch2coreml


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def validate_model_packages(args):
    """ Validate given model packages.
    """
    model_packages = args.model_packages

    if len(model_packages) < 2:
        raise RuntimeError("Require at least two models to merge")

    for model_package in model_packages:
        file_name = os.path.basename(model_package)
        if not file_name.lower().endswith("mlpackage"):
            raise RuntimeError("Expecting the given model packages to reference a `mlpackage`")


def validate_function_names(args, function_names):
    """ Verify that a function name exists for each model and they are all unique.
    """
    model_packages = args.model_packages
    default_function_name = args.default_function_name

    if len(model_packages) != len(function_names):
        raise RuntimeError("Expecting a function name for each model")

    if len(set(function_names)) != len(function_names):
        raise RuntimeError("Expecting all function names to be unique")

    if default_function_name is not None and default_function_name not in function_names:
        raise RuntimeError("Expecting the default function name to exist in given function names")


def merge(args):
    """ Merge models and return the path to the saved model.
    """
    def sanatize_function_name(name: str) -> str:
        """ Sanatize function name.
        """
        name = name.split("/")[-1]
        name = re.sub(r"[^\w]", "", name, flags=re.IGNORECASE)
        return name

    model_packages = args.model_packages
    function_names = args.function_names if args.function_names is not None else []
    is_extracting_function_names = len(function_names) == 0

    logger.info(f"Merging {len(model_packages)} models")

    desc = ct.utils.MultiFunctionDescriptor()
    for index, model_package in enumerate(model_packages):
        if is_extracting_function_names:
            model_spec = ct.utils.load_spec(model_package)
            if "lora_model" in model_spec.description.metadata.userDefined:
                function_name = model_spec.description.metadata.userDefined["lora_model"]
                function_name = sanatize_function_name(function_name)
            else:
                # If no metadata is found, default to `original` (assuming
                # this is the base model with no adapters).
                function_name = "original"
            function_names.append(function_name)
        else:
            function_name = function_names[index]

        logger.info(f"Adding {model_package} with function name `{function_name}`")
        desc.add_function(model_package, src_function_name="main", target_function_name=function_name)

    # Verify function names.
    validate_function_names(args, function_names)

    # Set the default function name
    default_function_name = args.default_function_name
    if default_function_name is None:
        default_function_name = function_names[0]
    desc.default_function_name = default_function_name
    logger.info(f"Setting default function to `{default_function_name}`")

    # Make output directory if it doesn't already exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
        logger.info(f"Created {args.output_path} for output")

    # Generate the filename for the merged model (`func1_func2_..._funcN.mlpackage`)
    merged_model_filename = "_".join(function_names) + ".mlpackage"
    output_model_package = os.path.join(args.output_path, merged_model_filename)

    logger.info(f"Saving merged model {output_model_package}")
    ct.utils.save_multifunction(desc, output_model_package)

    logger.info("Done")
    return output_model_package


def bundle_resources_for_swift_cli(args, output_model_package):
    """ Compiles the Core ML model from mlpackage into mlmodelc format.
    """
    resources_path = os.path.join(args.output_path, "Resources")
    model_name = "Unet.mlmodelc"

    # Create Resources folder
    if not os.path.exists(resources_path):
        os.makedirs(resources_path, exist_ok=True)
        logger.info(f"Created {resources_path} for Swift CLI assets")

    # Remove the existing U-net model if present
    full_model_path = os.path.join(resources_path, model_name)
    if os.path.exists(full_model_path):
        logger.warning(f"Overriding existing compiled model at {full_model_path}!")
        shutil.rmtree(full_model_path, ignore_errors=True)

    # Compile
    torch2coreml._compile_coreml_model(
        source_model_path=output_model_package,
        output_dir=resources_path,
        final_name=os.path.splitext(model_name)[0]
    )


def main(args):
    validate_model_packages(args)

    # Merge models
    output_model_package = merge(args)

    # Compile model
    if args.bundle_resources_for_swift_cli:
        bundle_resources_for_swift_cli(args, output_model_package)


def parser_spec():
    parser = argparse.ArgumentParser(
        usage=
        "merge.py --model-packages model1.mlpackage model2.mlpackage --output-path ./ "
        "--bundle-resources-for-swift-cli"
    )
    parser.add_argument(
        "--model-packages",
        nargs='+',
        type=str,
        required=True,
        help="A list of single function (`main`) model packages to merge."
    )
    parser.add_argument(
        "--function-names",
        nargs='+',
        type=str,
        required=False,
        help=
        "A list of function names associated with each model given in `--model-packages`. "
        "If no list is given, function names are derived using the user metadata field `lora_model`."
    )
    parser.add_argument(
        "--default-function-name",
        default=None,
        type=str,
        help=
        "The default function name from the given function names list, `--function-names`. "
        "If no default function is given, the first function in `--function-names` will be set to the default."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="The merged mlpackage will be saved into this directory."
    )
    parser.add_argument(
        "--bundle-resources-for-swift-cli",
        action="store_true",
        help=
        "If specified, creates a resources directory compatible with the sample Swift CLI and "
        "compiles and copies the merged model to the directory, **replacing** the current `Unet.mlmodelc` "
        "if exists.")
    return parser


if __name__ == "__main__":
    parser = parser_spec()
    args = parser.parse_args()
    main(args)