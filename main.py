import json
import math
import random
import sys
from pathlib import Path
from typing import Tuple, Any, NoReturn

import hydrus_api  # type: ignore
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import Qt
import matplotlib.pyplot as plt  # type: ignore
import scipy.stats as stats  # type: ignore
from trueskill import Rating, rate  # type: ignore
import numpy as np

DEFAULT_FILE_QUERY = ["system:number of tags > 5", "system:filetype = image", "system:limit = 5000"]
AMOUNT_OF_TAGS_IN_CHARTS = 20

FileMetaData = dict[str, Any]


class RatingSystem:
    def __init__(self, files_path: Path, client: hydrus_api.Client, file_ids: list[int]):
        self.files_path = files_path
        self.client = client
        self.file_ids = file_ids
        self.used_file_pairs: set[tuple[int, int]] = set()

        self.current_ratings: dict[str, Rating] = {}

        if Path("./ratings.json").exists():
            with open(Path("./ratings.json")) as f:
                tag_to_ratings = json.loads(f.read())
                for tag, rating_params in tag_to_ratings:
                    self.current_ratings[tag] = Rating(rating_params[0], rating_params[1])

        self.go_back_ratings_stack: list[dict[str, Rating]] = []

    def process_undo(self):
        try:
            last_ratings = self.go_back_ratings_stack.pop()
        except IndexError:
            return  # nothing to return to.

        for (tag, rating) in last_ratings.items():
            self.current_ratings[tag] = rating

    def write_results_to_file(self):
        with open(Path("./ratings.json"), "w") as f:
            f.write(json.dumps([(tag, [rating.mu, rating.sigma]) for tag, rating in self.current_ratings.items()]))

    def get_file_pair(self) -> None | Tuple[FileMetaData, FileMetaData]:
        ids: list[int] = random.sample(self.file_ids, k=2)

        tries = 0
        while tuple(ids) in self.used_file_pairs:
            if tries > 20:
                print("Tried to find a new random file pair 20 times, did not succeed.")
                return None
            ids = random.sample(self.file_ids, k=2)
            tries += 1

        # mypy here does not know that this list of 2 ints turns into a tuple of 2 ints.
        self.used_file_pairs.add(tuple(ids))  # type: ignore
        return self.convert_image_ids_to_file_meta_data(tuple(ids))  # type: ignore

    def convert_image_ids_to_file_meta_data(self, pairs: Tuple[int, int]) -> None | Tuple[FileMetaData, FileMetaData]:
        info = self.client.get_file_metadata(file_ids=pairs)
        if info is None:
            print(f"ERROR: Was not able to find the file metadata objects for ids '{pairs}'.")
            return None

        metadata = info["metadata"]
        if metadata is None:
            print(f"ERROR: The metadata object for the file pair '{pairs}' is None! (Maybe this script need to be updated?)")
            return None
        if not isinstance(metadata, list):
            print(f"ERROR: The metadata object for the file pair '{pairs}' is not a list! (Maybe this script needs to be updated?)")
            print(f"  This is what I did get: {metadata}")
            return None
        if len(metadata) != 2:
            print(f"ERROR: Did not get two metadata objects for the file pairs '{pairs}'.")
            print(f"  This is what I did get: {metadata}")
            return None

        # ignore the type here since mypy does not understand that we verified the type above.
        return tuple(metadata)  # type: ignore

    def path_from_metadata(self, file_1_metadata: FileMetaData) -> Path:
        file_hash = file_1_metadata["hash"]
        extension = file_1_metadata["ext"]

        return self.files_path / ("f" + file_hash[:2]) / (file_hash + extension)

    def process_result(self, *, winner: FileMetaData, loser: FileMetaData):
        winner_tags = self.tags_from_file(winner)
        loser_tags = self.tags_from_file(loser)

        winner_ratings = tuple([self.rating_for_tag(tag) for tag in winner_tags])
        loser_ratings = tuple([self.rating_for_tag(tag) for tag in loser_tags])

        # lower rank is better.
        new_winner_ratings, new_loser_ratings = rate([winner_ratings, loser_ratings], ranks=[0, 1])

        # first process loser then process winner, so that the tags that are in both images get the props for winning.
        # We may want to experiment with only updating tags that are not on both images?
        # though the issue there is that super common tags like 1girl would almost never get rated.
        # and you may also get super weird ratings for tags that are barely ever used.
        go_back_ratings: dict[str, Rating] = dict()
        for tag, new_rating in zip(loser_tags, new_loser_ratings):
            go_back_ratings[tag] = self.current_ratings[tag]
            self.current_ratings[tag] = new_rating

        for tag, new_rating in zip(winner_tags, new_winner_ratings):
            if tag not in loser_tags:  # otherwise we'd take the newly set value from the loser update here.
                go_back_ratings[tag] = self.current_ratings[tag]
            self.current_ratings[tag] = new_rating

        self.go_back_ratings_stack.append(go_back_ratings)

    # noinspection PyMethodMayBeStatic
    def tags_from_file(self, file: FileMetaData) -> list[str]:
        # dict of tag repos that may have some tag info.
        tag_repos: dict[str, dict[str, Any]] = file["tags"]
        tags: set[str] = set()
        for repo in tag_repos.values():
            if repo["display_tags"] is not None:
                if str(hydrus_api.TagStatus.CURRENT.value) in repo["display_tags"]:
                    tags.update(repo["display_tags"][str(hydrus_api.TagStatus.CURRENT)])

                if str(hydrus_api.TagStatus.PENDING.value) in repo["display_tags"]:
                    tags.update(repo["display_tags"][str(hydrus_api.TagStatus.PENDING)])

        # we need to go to list here since we need the ordering of this in keeping track of scores.
        return list(tags)

    def rating_for_tag(self, tag: str) -> Rating:
        if tag not in self.current_ratings:
            self.current_ratings[tag] = Rating()

        return self.current_ratings[tag]


class Window(QtWidgets.QWidget):
    def __init__(self, rating_system: RatingSystem):
        super().__init__()

        # these are set up in Window#perform_comparison_for_pair
        self.left_file_metadata: FileMetaData = {}
        self.right_file_metadata: FileMetaData = {}

        self.rating_system: RatingSystem = rating_system

        self.go_back_image_pairs_stack: list[Tuple[int, int]] = []
        self.comparisons = 0

        self.set_window_title_based_on_comparison_count()
        self.setLayout(QtWidgets.QHBoxLayout())

        self.leftImageLabel = QtWidgets.QLabel("left image")
        self.rightImageLabel = QtWidgets.QLabel("right image")

        self.layout().addWidget(self.leftImageLabel)
        self.layout().addWidget(self.rightImageLabel)

        for label in [self.leftImageLabel, self.rightImageLabel]:
            label.setMinimumWidth(500)
            label.setMinimumHeight(500)

        self.store_metadata_and_show_images_for_comparison_pair(self.rating_system.get_file_pair())

    def set_window_title_based_on_comparison_count(self):
        self.setWindowTitle(f"TagRank - Comparisons done this session: {self.comparisons}")

    def store_image_pair_onto_undo_stack(self, left_metadata: FileMetaData, right_metadata: FileMetaData):
        left_id = left_metadata["file_id"]
        right_id = right_metadata["file_id"]

        self.go_back_image_pairs_stack.append((left_id, right_id))

    def store_metadata_and_show_images_for_comparison_pair(self, metadatas: Tuple[FileMetaData, FileMetaData] | None):
        if metadatas is None:
            print("Was, for any reason, not able to load a pair of files. Shutting down now.")
            self.quit()
            return

        self.left_file_metadata, self.right_file_metadata = metadatas

        left_file_path = self.rating_system.path_from_metadata(self.left_file_metadata)
        right_file_path = self.rating_system.path_from_metadata(self.right_file_metadata)

        self.leftImageLabel.setPixmap(
            QtGui.QPixmap(left_file_path).scaled(self.leftImageLabel.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.FastTransformation))
        self.rightImageLabel.setPixmap(
            QtGui.QPixmap(right_file_path).scaled(self.rightImageLabel.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.FastTransformation))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.store_metadata_and_show_images_for_comparison_pair((self.left_file_metadata, self.right_file_metadata))

    def process_undo(self):
        try:
            image_ids = self.go_back_image_pairs_stack.pop()
        except IndexError:
            return  # nothing to go back to

        # we don't want to store metadata objects as they are quite large. So we as the client for them again.
        meta_datas = self.rating_system.convert_image_ids_to_file_meta_data(image_ids)

        # we need to make sure that the ratings are pulled back before the user can see the new images.
        self.rating_system.process_undo()
        self.store_metadata_and_show_images_for_comparison_pair(meta_datas)

        self.comparisons -= 1
        self.set_window_title_based_on_comparison_count()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Left:
            self.rating_system.process_result(winner=self.left_file_metadata, loser=self.right_file_metadata)
        elif key == QtCore.Qt.Key.Key_Right:
            self.rating_system.process_result(winner=self.right_file_metadata, loser=self.left_file_metadata)
        elif key == QtCore.Qt.Key.Key_Down:
            # print("No clear winner.")
            # TODO: Maybe we want to process draws as well? (TrueSkill supports that.)
            #       How does that influence the data?
            pass
        elif key == QtCore.Qt.Key.Key_Escape:
            self.quit()
            return
        elif key == QtCore.Qt.Key.Key_Backspace:
            self.process_undo()
            return  # return, since we don't want to move on to the next image pair below.
        else:  # ignore this event
            return

        self.comparisons += 1
        self.set_window_title_based_on_comparison_count()

        self.store_image_pair_onto_undo_stack(self.left_file_metadata, self.right_file_metadata)
        self.store_metadata_and_show_images_for_comparison_pair(self.rating_system.get_file_pair())

    def quit(self):
        print("Saving results to file...")
        self.rating_system.write_results_to_file()
        self.close()


def print_access_key_info_then_exit() -> NoReturn:
    print("  You need to create a client api service via services->review services->local->client api->add->manually")
    print("  It needs to have the permission search and fetch files.")
    print("  You can blacklist any tags you want, but they won't get ranked if this program cannot see them.")
    print("  When you have done this. Place the access key in a file called 'ACCESS_KEY' in the same folder as the main.py file.")
    print("  Then exit these windows by pressing apply.")
    print()

    print("  Now you need to turn on the client API.")
    print_enable_client_api_help()

    print()
    print("  If you have a non-standard URL or PORT you can place the url in a file called URL in the same folder as the main.py file.")
    print("  It should roughly follow the format of 'http://127.0.0.1:45869/'.")
    sys.exit(0)


def print_files_path_info_then_exit() -> NoReturn:
    print("  The FILES_PATH file is a file with name 'FILES_PATH' that needs to be in the same folder as the main.py file.")
    print("  The content of the file must be the full path to the folder in your hydrus installation that ends in client_files.")
    print("  It can for example look like this: '/home/user/Hydrus Network/db/client_files'.")
    print("  Or, on windows: 'C:\\Users\\user\\Hydrus Network\\db\\client_files'.")
    print()
    print("  The hydrus client can tell you where the files are by going to:")
    print("  Help -> About -> Description")
    print("  Then, somewhere near the bottom it says 'db dir: <PATH HERE>'.")
    print("  This is the exact path you should place in the FILES_PATH file.")
    sys.exit(0)


def print_verification_server_error_help_then_exit(e: None | hydrus_api.ServerError = None) -> NoReturn:
    print("ERROR: Something went wrong trying to verify your access key.")
    print("  Try re-creating your client api and saving the new access key. If need info on how. Remove the ACCESS_KEY file and restart TagRank.")
    if e is not None:
        print("  If that does not solve your issue, then look at the error that hydrus gave me below.")
        print("  Read it all, but the last line is probably where you'll find what is wrong.")
        print("This is what the server told me:")
        print(e)
    sys.exit(0)


def print_connection_error_help_then_exit(e: hydrus_api.ConnectionError) -> NoReturn:
    print("ERROR: Was not able to connect to hydrus.")
    print("  Are you sure your hydrus client is on?")
    print("  If it is, ensure that the API itself is on.")
    print_enable_client_api_help()
    print("  This is the error that caused the connection problem:")
    print(e)
    sys.exit(0)


def print_enable_client_api_help():
    print("  Go to Services -> Manage Services -> (double click) client api.")
    print("  Then ensure that the 'run the client api?' tick-box is on.")
    print("  Exit these windows by pressing apply.")


def print_permissions_error_then_exit(e: (hydrus_api.InsufficientAccess | None) = None) -> NoReturn:
    print("ERROR: This access key is not allowed to search for and fetch files.")
    print("  Please allow this permission for the access key you put in the ACCESS_KEY file.")
    print("  You can find this setting at: services->review services->local->client api")
    print()
    if e is not None:
        print("We know this because the client returned the following error: ")
        print(e)
    sys.exit(0)


def print_no_relevant_files_then_exit(query: list[str]) -> NoReturn:
    print(f"ERROR: Was not able to find enough files in the client to compare.")
    print(f"  Are you sure I am allowed to search for files?")
    print(f"  I am specifically searching for files that are found by searching for the following query:")
    print(f"  {', '.join(query)}")
    print(f"  If this query looks weird, change it in the SEARCH_QUERY file.")
    sys.exit(0)


def print_search_query_help():
    print("The search query file (SEARCH_QUERY) has just been made, and populated with the default query.")
    print("Every line of this file is used as one 'tag' to search your client.")
    print("You can do quite advanced things with this search. See the API documentation for more info.")
    print("https://hydrusnetwork.github.io/hydrus/developer_api.html#get_files_search_files")
    print("Scroll down a little to the `system predicates` expando to see examples of system queries you can do.")


def print_empty_query_help_then_exit() -> NoReturn:
    print("ERROR: the file query is empty.")
    print("Since this may lead to very large queries, this is not allowed.")
    print("If you really want the search to return all files, add 'system: everything' to the SEARCH_QUERY file.")
    print("If you want to return to the default search query delete the SEARCH_QUERY file.")
    print("It will be remade with the default query when you start this script again.")
    sys.exit(0)


def trueskill_number_from_rating(rating: Rating) -> float:
    return rating.mu - (3*rating.sigma)


def main() -> None:
    key_path = Path("./ACCESS_KEY")
    if not key_path.exists():
        print("ERROR: ACCESS_KEY file does not exist.")
        print_access_key_info_then_exit()

    access_key = key_path.read_text()
    if access_key == "":
        print("ERROR: ACCESS_KEY file is empty.")
        print_access_key_info_then_exit()

    access_key = access_key.removesuffix("\n")

    url_path = Path("./URL")
    if url_path.exists():
        url: str | None = url_path.read_text()
        if url == "":
            url = None
    else:
        url = None

    if url is not None:
        client = hydrus_api.Client(access_key, api_url=url)
    else:
        client = hydrus_api.Client(access_key)

    access_key_response = None
    try:
        access_key_response = client.verify_access_key()
    except hydrus_api.ServerError as e:
        print_verification_server_error_help_then_exit(e)
    except hydrus_api.ConnectionError as e:
        print_connection_error_help_then_exit(e)
    except hydrus_api.InsufficientAccess as e:
        print_permissions_error_then_exit(e)

    if access_key_response is None:
        print_verification_server_error_help_then_exit()

    if 3 not in access_key_response["basic_permissions"]:
        print_permissions_error_then_exit(None)

    files_path_path = Path("./FILES_PATH")
    if not files_path_path.exists():
        print("ERROR: FILES_PATH file does not exist.")
        print_files_path_info_then_exit()

    files_path_text = files_path_path.read_text()
    if files_path_text == "":
        print("ERROR: FILES_PATH file is empty.")
        print_files_path_info_then_exit()

    clean_path_text = files_path_text.removesuffix("\n").removesuffix("\\").removesuffix("/")

    files_path = Path(clean_path_text)

    # "f00" is one of the folders that the files are actually in.
    if not (files_path / "f00").exists():
        # files path does not exist. Did the user forgot this postfix?
        if not clean_path_text.endswith("client_files"):
            files_path = files_path / "client_files"

        if not (files_path / "f00").exists():
            print(f"ERROR: The files path '{Path(clean_path_text).resolve()}' does not exist.")
            print_files_path_info_then_exit()

    if not files_path.is_dir():
        print(f"ERROR: the files path '{files_path}' is not a directory.")
        print_files_path_info_then_exit()

    file_query_path = Path("./SEARCH_QUERY")
    if not file_query_path.exists():
        file_query_path.write_text("\n".join(DEFAULT_FILE_QUERY))
        print_search_query_help()

    if file_query_path.read_text().strip() == "":
        print_empty_query_help_then_exit()


    if file_query_path.read_text().strip() == """
system:number of tags > 5
system:filetype = image
system:limit = 500""".strip():
        print("You where using the previous default file_query. It has been updated to the following:")
        print("\n".join(DEFAULT_FILE_QUERY))
        file_query_path.write_text("\n".join(DEFAULT_FILE_QUERY))


    query = list(filter(lambda s: s != "", file_query_path.read_text().splitlines()))

    relevant_files_ids = client.search_files(query, file_sort_type=hydrus_api.FileSortType.RANDOM)
    print("THIS IS THE DEBUG INFO RIGHT HERE:")

    print("from API:")
    print(type(relevant_files_ids))
    print(relevant_files_ids)

    print("raw:")

    params = {"tags": json.dumps(query, cls=hydrus_api._ABCJSONEncoder)}
    params["file_sort_type"] = hydrus_api.FileSortType.RANDOM

    response = client._api_request("GET", client._SEARCH_FILES_PATH, params=params)

    print(f"status code: {response.status_code}")
    print("Response text:")
    print(response.text)
    print("THIS WAS THE DEBUG INFO.")

    if relevant_files_ids is None or relevant_files_ids["file_ids"] is None or len(relevant_files_ids["file_ids"]) < 2:
        print_no_relevant_files_then_exit(query)

    app = QtWidgets.QApplication(sys.argv)
    rating_system = RatingSystem(files_path, client, relevant_files_ids["file_ids"])
    window: QtWidgets.QWidget = Window(rating_system)

    window.show()
    first_section_result = app.exec()
    if first_section_result != 0:
        print("Comparison app closed in error. Not moving on to comparisons.")
        sys.exit(first_section_result)
    window.destroy()

    many_tags: list[Tuple[str, Rating]] = sorted(rating_system.current_ratings.items(),
                                                 key=lambda x: trueskill_number_from_rating(x[1]),
                                                 reverse=True)[:max(100, AMOUNT_OF_TAGS_IN_CHARTS)]

    largest_mu_width = len(str(math.floor(trueskill_number_from_rating(many_tags[0][1]))))
    print("The window that shows the scores can be hard to read. So here the data in text for 100 tags:")
    for (tag, rating) in many_tags:
                                                                # +3 for the three decimals
        print(f"{trueskill_number_from_rating(rating):.3f}".rjust(largest_mu_width+3) + f": {tag}")

    best_tags: list[Tuple[str, Rating]] = many_tags[:AMOUNT_OF_TAGS_IN_CHARTS]

    for (tag, rating) in best_tags:
        (mu, sigma) = rating
        x_space = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y_space = stats.norm.pdf(x_space, mu, sigma)
        plt.plot(
            x_space,
            y_space,
            label=f"{tag} (score:{trueskill_number_from_rating(rating):.2f})"
        )

    plt.legend()  # show a legend
    plt.show()

    # TODO: Choose files to play against each other. Maybe use some halfway point between high and low win prob?
    #       Or use files where win prob is ~50% so that we get "new" info

    # TODO: Test between (not) including duplicate tags in the scoring.
    #       How does this affect the scoring tags?
    #       Will super common tags stay in the middle since they aren't played very often?
    #       Maybe this will happen regardless since they win and loose as commonly.


if __name__ == "__main__":
    main()
