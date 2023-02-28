import json
import random
import sys
from pathlib import Path
from typing import Tuple, Any, NoReturn

import hydrus_api  # type: ignore
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import Qt
from trueskill import Rating, rate  # type: ignore

DEFAULT_FILE_QUERY = ["system:number of tags > 5", "system:filetype = image", "system:limit = 500"]

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

        info = self.client.get_file_metadata(file_ids=ids)
        if info is None:
            print(f"ERROR: Was not able to find the file metadata objects for ids '{ids}'.")
            return None

        metadata = info["metadata"]
        if metadata is None:
            print(f"ERROR: The metadata object for the file pair '{ids}' is None! (Maybe this script need to be updated?)")
            return None
        if not isinstance(metadata, list):
            print(f"ERROR: The metadata object for the file pair '{ids}' is not a list! (Maybe this script needs to be updated?)")
            print(f"  This is what I did get: {metadata}")
            return None
        if len(metadata) != 2:
            print(f"ERROR: Did not get two metadata objects for the file pairs '{ids}'.")
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
        loser_ratings  = tuple([self.rating_for_tag(tag) for tag in loser_tags])

        new_winner_ratings, new_loser_ratings = rate([winner_ratings, loser_ratings])

        # first process loser then process winner, so that the tags that are in both images get the props for winning.
        # We may want to experiment with only updating tags that are not on both images?
        # though the issue there is that super common tags like 1girl would almost never get rated.
        # and you may also get super weird ratings for tags that are barely ever used.
        for tag, new_rating in zip(loser_tags, new_loser_ratings):
            self.current_ratings[tag] = new_rating

        for tag, new_rating in zip(winner_tags, winner_ratings):
            self.current_ratings[tag] = new_rating


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
        if tag in self.current_ratings:
            return self.current_ratings[tag]

        return Rating()


class Window(QtWidgets.QWidget):
    def __init__(self, rating_system: RatingSystem):
        super().__init__()

        # these are set up in Window#perform_comparison_for_pair
        self.left_file_metadata: FileMetaData = {}
        self.right_file_metadata: FileMetaData = {}

        self.rating_system: RatingSystem = rating_system

        self.setWindowTitle("TagRank")
        self.setLayout(QtWidgets.QHBoxLayout())

        self.leftImageLabel = QtWidgets.QLabel("left image")
        self.rightImageLabel = QtWidgets.QLabel("right image")

        self.layout().addWidget(self.leftImageLabel)
        self.layout().addWidget(self.rightImageLabel)

        for label in [self.leftImageLabel, self.rightImageLabel]:
            label.setMinimumWidth(500)
            label.setMinimumHeight(500)

        self.perform_comparison_for_pair(self.rating_system.get_file_pair())


    def perform_comparison_for_pair(self, metadatas: Tuple[FileMetaData, FileMetaData] | None):
        if metadatas is None:
            print("Was, for any reason, not able to load a pair of files. Shutting down now.")
            self.quit()
            return

        self.left_file_metadata, self.right_file_metadata = metadatas

        left_file_path = self.rating_system.path_from_metadata(self.left_file_metadata)
        right_file_path = self.rating_system.path_from_metadata(self.right_file_metadata)

        self.leftImageLabel.setPixmap(QtGui.QPixmap(left_file_path).scaled(self.leftImageLabel.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        self.rightImageLabel.setPixmap(QtGui.QPixmap(right_file_path).scaled(self.rightImageLabel.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))


    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Left:
            self.rating_system.process_result(winner = self.left_file_metadata, loser = self.right_file_metadata)
        elif key == QtCore.Qt.Key.Key_Right:
            self.rating_system.process_result(winner = self.right_file_metadata, loser = self.left_file_metadata)
        elif key == QtCore.Qt.Key.Key_Down:
            # print("No clear winner.")
            pass
        elif key == QtCore.Qt.Key.Key_Escape:
            self.quit()
            return
        else: # ignore this event
            return

        self.perform_comparison_for_pair(self.rating_system.get_file_pair())


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
    print("  Now you need to start the api via the services->manage services->double click 'client api'->un-select 'do not run client api service'.")
    print("  Then exit these windows by pressing apply.")
    print("  If you have a non-standard URL or PORT you can place the url in a file called URL in the same folder as the main.py file.")
    print("  It should roughly follow the format of 'http://127.0.0.1:45869/'.")
    sys.exit(0)


def print_files_path_info_then_exit() -> NoReturn:
    print("  The FILES_PATH file is a file with name 'FILES_PATH' that needs to be in the same folder as the main.py file.")
    print("  The content of the file must be the full path to the folder in your hydrus installation that ends in client_files.")
    print("  It can for example look like this: '/home/user/Hydrus Network/db/client_files'.")
    print("  Or, on windows: 'C:\\Users\\user\\Hydrus Network\\db\\client_files'.")
    sys.exit(0)


def print_verification_server_error_help_then_exit(e: None | hydrus_api.ServerError = None) -> NoReturn:
    print("ERROR: Something went wrong trying to verify your access key.")
    print("  Try re-creating your client api and saving the new access key. If need info on how. Remove the ACCESS_KEY file and restart TagRank.")
    if e is not None:
        print("  If that does not solve your issue, then look at the error that hydrus gave me below.")
        print("  Read it all, but the last line is probably where you'll find what is wrong.")
        print("This is what the server told me:")
        print(e)
    sys.exit(1)


def print_permissions_error_then_exit() -> NoReturn:
    print("ERROR: This access key is not allowed to search for and fetch files.")
    print("  Please allow this permission for the access key you put in the ACCESS_KEY file.")
    print("  You can find this setting at: services->review services->local->client api")
    sys.exit(1)


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
    print("Scroll down a little to the system predicated expando to see examples of system queries you can do.")


def main():
    key_path = Path("./ACCESS_KEY")
    if not key_path.exists():
        print("ERROR: ACCESS_KEY file does not exist.")
        print_access_key_info_then_exit()

    access_key = key_path.read_text()
    if access_key == "":
        print("ERROR: ACCESS_KEY file is empty.")
        print_access_key_info_then_exit()

    url_path = Path("./URL")
    if url_path.exists():
        url = url_path.read_text()
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

    if access_key_response is None:
        print_verification_server_error_help_then_exit()

    if 3 not in access_key_response["basic_permissions"]:
        print_permissions_error_then_exit()

    files_path_path = Path("./FILES_PATH")
    if not files_path_path.exists():
        print("ERROR: FILES_PATH file does not exist.")
        print_files_path_info_then_exit()

    files_path_text = files_path_path.read_text()
    if files_path_text == "":
        print("ERROR: FILES_PATH file is empty.")
        print_files_path_info_then_exit()

    files_path = Path(files_path_text)
    if not files_path.exists():
        print(f"ERROR: The files path '{files_path}' does not exist.")
        print_files_path_info_then_exit()

    if not files_path.is_dir():
        print(f"ERROR: the files path '{files_path}' is not a directory.")
        print_files_path_info_then_exit()

    file_query_path = Path("./SEARCH_QUERY")
    if not file_query_path.exists():
        file_query_path.write_text("\n".join(DEFAULT_FILE_QUERY))
        print_search_query_help()

    query = list(filter(lambda s: s != "", file_query_path.read_text().splitlines()))

    relevant_files_ids = client.search_files(query, file_sort_type=hydrus_api.FileSortType.RANDOM)
    if relevant_files_ids is None or relevant_files_ids["file_ids"] is None or len(relevant_files_ids["file_ids"]) < 2:
        print_no_relevant_files_then_exit(query)

    app = QtWidgets.QApplication(sys.argv)
    window = Window(RatingSystem(files_path, client, relevant_files_ids["file_ids"]))
    window.show()
    sys.exit(app.exec())


    # TODO: Choose files to play against each other. Maybe use some halfway point between high and low win prob?
    #       Or use files where win prob is ~50% so that we get "new" info

    # TODO: Test between (not) including duplicate tags in the scoring.
    #       How does this affect the scoring tags? Will super common tags stay in the middle since they aren't played very often?
    #       Maybe this will happen regardless since they win and loose as commonly.

    # TODO: Make nice matplotlib chart of scores?


if __name__ == "__main__":
    main()
