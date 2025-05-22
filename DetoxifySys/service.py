from detoxify import Detoxify
import json
import argparse


def censor_comment(text, threshold=0.5):
    """
    Analyze a single comment for toxicity using Detoxify multilingual model.
    Args:
        text (str): Comment to analyze.
        threshold (float): Threshold for deciding if comment is inappropriate.
    Returns:
        dict: Toxicity scores and decision.
    """
    model = Detoxify("multilingual")
    results = model.predict([text])
    scores = {k: round(v[0], 5) for k, v in results.items()}
    # Decide if any toxicity score exceeds the threshold
    is_appropriate = all(score < threshold for score in scores.values())
    decision = "appropriate" if is_appropriate else "inappropriate"
    return {"scores": scores, "decision": decision}


def cli_main():

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--text",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        default=0.5,
        type=float,
        help="threshold for deciding if comment is inappropriate (default: 0.5)",
    )
    res = censor_comment(
        text=parser.parse_args().text,
        threshold=parser.parse_args().threshold,
    )
    json_object = json.dumps(res, indent=4)

    # Writing to sample.json
    with open("DetoxifySys/assets/result.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    cli_main()
