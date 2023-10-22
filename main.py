import contextlib
import itertools
import json
import math
import os
import random
import sys
from functools import cmp_to_key
from importlib.metadata import version
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, Any, NoReturn

import hydrus_api  # type: ignore
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import Qt
import matplotlib.pyplot as plt  # type: ignore
import scipy.stats as stats  # type: ignore
from trueskill import Rating, rate, BETA, global_env  # type: ignore
import numpy as np

h_api_version = version('hydrus_api')

if h_api_version is None:
    # cannot check version for some reason.
    pass
elif len(h_api_version.split(".")) < 3:
    # Version is in a weird format. Ignore.
    pass
else:
    try:
        major: str
        minor: str
        patch: str

        major, minor, patch = h_api_version.split(".")
        if int(major) < 5:
            print("Your hydrus_api version is not up to date!")
            print(f"Tagrank is seeing version {h_api_version}, but requires at least version 5.0.0.")
            print("You can update your hydrus_api version with the command `pip install --upgrade hydrus_api`.")
            print("If you have done so, tagrank is up to date, and this error still comes up please make a report on github or on discord.")
            print("Be sure to include the output of `pip freeze` and the error message you are now reading.")
            sys.exit(1)

    except ValueError:
        # failed to unpack. Ignore.
        pass

    # we could do more with the minor or patch versions as well,
    # and then build up some table of compatible `hydrus_api`, `hydrus`, and `tagrank` versions.
    # that does not seem worth the effort for now, but if we get a lot more issues like this we may do so.

DEFAULT_FILE_QUERY = ["system:number of tags > 5", "system:filetype = image", "system:limit = 5000"]
AMOUNT_OF_TAGS_IN_CHARTS = 20

FileMetaData = dict[str, Any]


def tags_from_file(file: FileMetaData) -> list[str]:
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

        # where the winner is the first of the two file ids
        self.known_comparison_choices: list[Tuple[int, int]] = []

        if Path("./comparisons.json").exists():  # if not exists, will be made on exit.
            try:
                with open(Path("./comparisons.json")) as f:
                    comparisons = json.loads(f.read())
                    for winner, loser in comparisons:
                        self.known_comparison_choices.append((winner, loser))
            except (JSONDecodeError, ValueError) as e:
                print_could_not_read_comparisons_file_help()
                raise e

    def process_undo(self):
        try:
            last_ratings = self.go_back_ratings_stack.pop()

            # if the above pop throws this will not happen.
            # This is good, since it ensures that we do not remove comparisons not made in this session,
            self.known_comparison_choices.pop()
        except IndexError:
            return  # nothing to return to.

        for (tag, rating) in last_ratings.items():
            self.current_ratings[tag] = rating

    def write_results_to_file(self):
        with open(Path("./ratings.json"), "w") as f:
            f.write(json.dumps([(tag, [rating.mu, rating.sigma]) for tag, rating in self.current_ratings.items()]))

        with open(Path("./comparisons.json"), "w") as f:
            f.write(json.dumps([[first, second] for first, second in self.known_comparison_choices]))

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
        winner_tags = tags_from_file(winner)
        loser_tags = tags_from_file(loser)

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

        self.known_comparison_choices.append((winner["file_id"], loser["file_id"]))

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
            self.exit()
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
        if key == QtCore.Qt.Key.Key_Left or key == QtCore.Qt.Key.Key_A:
            self.rating_system.process_result(winner=self.left_file_metadata, loser=self.right_file_metadata)
        elif key == QtCore.Qt.Key.Key_Right or key == QtCore.Qt.Key.Key_D:
            self.rating_system.process_result(winner=self.right_file_metadata, loser=self.left_file_metadata)
        elif key == QtCore.Qt.Key.Key_Down or key == QtCore.Qt.Key.Key_S:
            # print("No clear winner.")
            # TODO: Maybe we want to process draws as well? (TrueSkill supports that.)
            #       How does that influence the data?
            pass
        elif key == QtCore.Qt.Key.Key_Escape:
            self.exit()
            return
        elif key == QtCore.Qt.Key.Key_Backspace or key == QtCore.Qt.Key.Key_R:
            self.process_undo()
            return  # return, since we don't want to move on to the next image pair below.
        elif key == QtCore.Qt.Key.Key_O:
            self.open_files_externally()
            return  # return, since we don't want to move on to the next image pair below.
        else:  # ignore this event
            return

        self.comparisons += 1
        self.set_window_title_based_on_comparison_count()

        self.store_image_pair_onto_undo_stack(self.left_file_metadata, self.right_file_metadata)
        self.store_metadata_and_show_images_for_comparison_pair(self.rating_system.get_file_pair())

    def open_files_externally(self) -> None:
        # user asked us to open these files in another program.
        file_path_right = "file://" + str(self.rating_system.path_from_metadata(self.right_file_metadata).resolve())
        file_path_left = "file://" + str(self.rating_system.path_from_metadata(self.left_file_metadata).resolve())

        try:
            # only available on windows. wew
            os.startfile(file_path_left)
            os.startfile(file_path_right)
        except AttributeError:
            # does not always work, so we try the python way first.
            with contextlib.redirect_stdout:
                # need to redirect since some browsers (Vivaldi, and thus I assume chromium)
                # will print which browser "session" they open in for each file.
                # cool information, but not relevant for our user.

                QtGui.QDesktopServices.openUrl(file_path_left)
                QtGui.QDesktopServices.openUrl(file_path_right)

    def exit(self) -> None:
        self.close()  # calls the close event, which will save the results to file

    def closeEvent(self, event) -> None:
        # this is called by self.close(), and when the window is closed by Qt in any other way.
        self.prepare_to_quit()

    def prepare_to_quit(self):
        print("Saving results to file...")
        self.rating_system.write_results_to_file()


def print_could_not_read_comparisons_file_help() -> None:
    print(f"ERROR: Was not able to read your comparisons.json file!")
    print(f"  The reason for this will be printed above, or below this information.")
    print(f"  If you do not know what the reason means you should do the following:")
    print(f"  1. Rename the file {Path('./comparisons.json').resolve()} to something else.")
    print(f"  2. Show the error and the file to me in the hydrus discord if you want to recover the comparisons.")
    print(f"  3. Re-open TagRank, it will start your comparisons list from new.", flush=True)


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


def print_could_not_fetch_file_information_then_exit() -> NoReturn:
    print("ERROR: Was not able to fetch file information.")
    print("  Are you sure that I have all the needed permissions?")
    sys.exit(0)


def print_no_relevant_files_to_sort_then_exit() -> NoReturn:
    print("ERROR: Was not able to find any files to sort.")
    print("  Are you sure you have any ranked tags?")
    print("  If so, are you sure that TagRank is allowed to search for files?")
    print("  If so, please report this error to me.")
    sys.exit(0)


def print_add_tags_permissions_missing_info_then_exit() -> NoReturn:
    print("ERROR: TagRank is not allowed to add tags to the client!")
    print("  In order to add the ranking tags to the client TagRank needs the 'edit file tags' permission.")
    print("  You can set this up by going to the following:")
    print("  Services -> Review Services -> local -> client api")
    print("  In this window, select the TagRank client api, then press 'edit' at the bottom of the screen.")
    print("  Now, in this window, check the checkbox before 'edit file tags'.")
    print("  Exit the window by pressing 'apply', then press 'close' to close the review services window.")
    print("  After you've done that, re-run TagRank.")
    sys.exit(0)


def trueskill_number_from_rating(rating: Rating) -> float:
    return rating.mu - (3*rating.sigma)


def create_client_or_exit() -> hydrus_api.Client:
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

    return client


def run_for_rank_tags(client) -> None:
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


def compare_two_teams(left_file: Tuple[int, list[Rating]], right_file:Tuple[int, list[Rating]]) -> int:
    left_team = left_file[1]
    right_team = right_file[1]
    p = win_probability(left_team, right_team)

    # p is in (0..1), where 1 means left team has 100% chance of winning.
    # Since left < right means we need to return negative, we can do that with -0.5
    # This means that p > 0.5 (left would win) returns >0, and draw, p=0.5, returns 0.
    return p - 0.5


# taken from issue #1 on the trueskill repo. It is also provided on their site.
def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
    ts = global_env()
    return ts.cdf(delta_mu / denom)


def delete_existing_sort_tags_if_needed(client: hydrus_api.Client) -> None:
    response = client.search_files(tags=["TagRankSort:*"])
    if response is None or response["file_ids"] is None:
        print("I was not able to search for files or something went wrong when trying to.")
        print("Please check your permissions with the following help text.")
        print("If this does not help please report this error.")
        print_permissions_error_then_exit(None)

    if len(response["file_ids"]) == 0:
        return

    print("You still have files with the TagRankSort tags from an earlier sort attempt!")

    still_has_tags_response = client.get_file_metadata(file_ids=response["file_ids"])
    if still_has_tags_response is None or still_has_tags_response["metadata"] is None:
        print("ERROR: Was not able to get the data to delete the existing ranking tags.")
        print("  Are you sure I have the required permissions?")
        print("  Otherwise, is TagRank maybe out of date compared to your hydrus version?")
        sys.exit(0)

    for metadata in still_has_tags_response["metadata"]:
        file_id = metadata["file_id"]
        for (tag_repo_identifier, tag_repo_data) in metadata["tags"].items():
            if "0" not in tag_repo_data["display_tags"]:
                continue

            previous_sort_tags = [tag for tag in tag_repo_data["display_tags"]["0"] if
                                  tag.startswith("TagRankSort:")]
            if len(previous_sort_tags) > 0:
                client.add_tags(file_ids=[file_id], service_keys_to_actions_to_tags={
                    tag_repo_identifier: {hydrus_api.TagAction.DELETE: previous_sort_tags}})

    print("Existing sort tags deleted.")


def run_for_create_image_ranking(client: hydrus_api.Client) -> None:
    if hydrus_api.Permission.ADD_TAGS not in client.verify_access_key()["basic_permissions"]:
        print_add_tags_permissions_missing_info_then_exit()

    delete_existing_sort_tags_if_needed(client)

    #  1. Find all images that have at least one of the scored tags.
    rating_system = RatingSystem(Path("."), client, [])
    tags = list(rating_system.current_ratings.keys())

    # The type does not include the "or search" system. Any nested list of tags is seen as OR.
    # noinspection PyTypeChecker
    response = client.search_files(tags=[tags])

    if response is None or response["file_ids"] is None or len(response["file_ids"]) == 0:
        print_no_relevant_files_to_sort_then_exit()

    file_ids = [int(file_id) for file_id in response["file_ids"]]

    print(f"Found {len(file_ids)} files that have at least one ranked tag.")

    file_infos_response = client.get_file_metadata(file_ids=file_ids)
    if file_infos_response is None or file_infos_response["metadata"] is None:
        print_could_not_fetch_file_information_then_exit()

    file_ids_to_tags: list[Tuple[int, list[str]]] = [(metadata["file_id"], tags_from_file(metadata)) for metadata in file_infos_response["metadata"]]

    print("Got the tags for each file from the client.")

    file_ids_to_ratings: list[Tuple[int, list[Rating]]] = [(file_id, [rating_system.rating_for_tag(tag) for tag in tags]) for (file_id, tags) in file_ids_to_tags]

    print("Now sorting the list... This may take a very long time!")
    #  2. Sort the list using the 1v1 win probability.
    # Note that we pass in reverse is true, since otherwise the worst item would be first.
    sorted_file_ids_to_ratings = sorted(file_ids_to_ratings, key=cmp_to_key(compare_two_teams), reverse=True)

    print("Sorted the list. Now setting the sort-order tags in hydrus.")

    services_response = client.get_services()

    services_map = services_response["services"]

    found_service_id = None
    for service_id, service_data in services_map.items():
        if service_data["type"] == hydrus_api.ServiceType.TAG_DOMAIN:
            if found_service_id is None:
                found_service_id = service_id
            if service_data["name"] == "my tags":
                found_service_id = service_id

    for (index, (file_id, _)) in enumerate(sorted_file_ids_to_ratings):
        client.add_tags(file_ids=[file_id], service_keys_to_tags={found_service_id: [f"TagRankSort:{index}"]})

    print("Have sent all the tags to the client.")
    print("DONE! If you need info on how to use this to sort your files, read below:")
    print("  You can use this sort order by clicking the 'sort by(...)' button on the top left of a file search column. ")
    print("  Here, select Namespaces -> Custom. Then fill in 'TagRankSort'. Press ok, select 'display tags'.")
    print("  If you want to make this easier, go to: file -> options -> sort/collect.")
    print("  In the 'namespace file sorting' section press 'add' at the bottom.")
    print("  Fill in 'TagRankSort', press ok, then select 'display tags'.")
    print("  Press apply to save these settings.")
    print("  Now, if you want to set this as the default sort: go to: file -> options -> sort/collect.")
    print("  Click the first button to the right of the text 'Default File Sort'")
    print("  Here, select Namespaces, and click the 'sort by tags: TagRankSort' option that you just created.")


def main(mode: str) -> None:
    client = create_client_or_exit()

    if mode == MODE_RANK_TAGS:
        run_for_rank_tags(client)
    elif mode == MODE_CREATE_IMAGE_RANKING:
        run_for_create_image_ranking(client)
    else:
        print("ERROR: Unknown run mode!")


MODE_CREATE_IMAGE_RANKING = "create_image_ranking"
MODE_RANK_TAGS = "rank_tags"

if __name__ == "__main__":
    if sys.argv:
        arguments = sys.argv
    else:
        arguments = []

    if "--create_image_ranking" in arguments:
        mode = MODE_CREATE_IMAGE_RANKING
    else:
        mode = MODE_RANK_TAGS

    main(mode)
