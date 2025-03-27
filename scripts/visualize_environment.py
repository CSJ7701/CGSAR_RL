import argparse
from simulation.Visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize an ocean environment with floating victims from an HDF5 file.")
    parser.add_argument("-d", "--data_file", type=str, required=True, help="Path to the HDF5 data file.")
    parser.add_argument("-s", "--save", type=bool, help="Save the animation as an mp4 file.")

    args = parser.parse_args()
    show = not args.save

    v=Visualizer(args.data_file)
    v.run(show=show)

