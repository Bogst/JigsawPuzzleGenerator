import argparse
from PuzzleConfigs import PuzzleConfigs
from generator import generate_puzzles


def pair(arg):
    return [int(x) for x in arg.split(',')]


def dir_path(path):
    import os
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def main():
    parser = argparse.ArgumentParser(description="Jigsaw puzzle generator.")
    parser.add_argument("--nr_puzzles", type=int, required=True, help="Number of puzzles to be generated")
    parser.add_argument('--input', type=dir_path, required=True, help="Path towards the folder containing the "
                                                                      "backgrounds and the puzzle faces")
    parser.add_argument('--output', type=dir_path, required=True, help="Path towards the folder where the puzzles"
                                                                       "should be created")
    parser.add_argument('--dimension', type=pair, default="2,2", help="(x,y) number of rows x and number of columns y "
                                                                      "of the jigsaw puzzle")
    parser.add_argument('--consideration_cutoff', type=int, default=2, help="number of control points on the edge to be"
                                                                            "ignored at both ends (done in order to "
                                                                            "take only the notch control points in "
                                                                            "consideration and not include the edge "
                                                                            "defining points)")
    parser.add_argument('--skip_gaussian', default=False, action="store_true", help="Skip applying the gaussian blur"
                                                                                    "after placing the pieces on the "
                                                                                    "background image")
    parser.add_argument('--blur_radius', default=2, type=int, help="The kernel size of the gaussian blur")
    parser.add_argument('--show_solution', default=False, action="store_true", help="Show solution on the shapes image")
    parser.add_argument('--log_to_screen', default=False, action="store_true", help="Log messages to screen")

    args = parser.parse_args()

    puzzle_configs = PuzzleConfigs(
        num_rows=args.dimension[0],
        num_cols=args.dimension[1],
        solution_consideration_cutoff=args.consideration_cutoff,
        apply_gaussian_blur=not args.skip_gaussian,
        gaussian_blur_radius=args.blur_radius,
        show_solution=args.show_solution,
        log_to_screen=args.log_to_screen
    )

    generate_puzzles(nr_puzzles=args.nr_puzzles, path=args.input, save_path=args.output, puzzle_configs=puzzle_configs)


if __name__ == "__main__":
    main()
