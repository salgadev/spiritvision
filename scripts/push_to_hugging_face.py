import argparse

from fastai.learner import load_learner

from helpers import get_models_dir, get_processed_data_dir, make_data_loader, resnet_learner

from huggingface_hub import push_to_hub_fastai
import os

repo_id = "artificeresearch/spiritvision"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m',
                        required=True,
                        help="Model name from the /models/ directory")

    args = parser.parse_args()

    model_name = str(args.model)
    load_path = os.path.join(get_models_dir(), model_name)

    data_loader = make_data_loader(get_processed_data_dir(), batch_size=64)

    learn = resnet_learner(data_loader, 50)
    learn = load_learner(load_path)
    push_to_hub_fastai(learner=learn, repo_id=repo_id)


if __name__ == "__main__":
    main()
