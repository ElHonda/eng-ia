import argparse

from eng_ia.agent import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent eng-ia")
    parser.add_argument("pergunta", nargs="?", help="Pergunta em português")
    args = parser.parse_args()
    run(args.pergunta)


if __name__ == "__main__":
    main()
