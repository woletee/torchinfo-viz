import cProfile
import pstats
import random
import torchvision
from tqdm import trange
from torchinfo import summary


def profile() -> None:
    """Prints top N methods, sorted by time."""
    random.seed(0)
    command = "for _ in trange(10): summary(torchvision.models.resnet152(), (1, 3, 224, 224), verbose=0)"
    profile_file = "profile.txt"
    sort = "time"
    cProfile.run(command, filename=profile_file, sort=sort)
    stats = pstats.Stats(profile_file)
    stats.sort_stats(sort).print_stats(50)


if __name__ == "__main__":
    profile()
