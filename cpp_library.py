from common import *
import os


class CppLibrary:
    def include_name(self):
        """
        :return: <iostream> <cmath>, or "user_lib.h"...
        """
        assert False, "must return a string"

    def __hash__(self):
        return hash(self.include_name())

    def __eq__(self, other: "CppLibrary"):
        return self.include_name() == other.include_name()


class UserLibrary(CppLibrary):
    _global_head_comments_h = ["// Copy right: Huaiyuan Teng.",
                             "// This file is auto generated, please don't modify."]
    _global_tail_comments_h = []

    _header_begin_comment_h = "//======header begin======"
    _header_end_comment_h = "//======header end======"

    _from_file_location_to_project_root_path = ""
    _project_root_abs_path = os.path.join(os.path.dirname(__file__), _from_file_location_to_project_root_path)

    @classmethod
    def set_global_project_root(cls, project_root: str):
        cls._project_root_abs_path = project_root

    @classmethod
    def global_project_root(cls):
        return cls._project_root_abs_path

    def head_comments_h(self)->List[str]:
        return self._global_head_comments_h

    def tail_comments_h(self)->List[str]:
        return self._global_tail_comments_h

    def header_begin_comments_h(self):
        return self._header_begin_comment_h

    def header_end_comments_h(self):
        return self._header_end_comment_h


    def __init__(self, relative_path: str, lib_name: str, project_root: str = _project_root_abs_path):
        """
        :param relative_path:
        :param lib_name:
        :param project_root:
        """

        self._project_root = project_root
        self._lib_name = lib_name
        self._relative_path = remove_front_slashes(relative_path)

    def include_name(self):
        return "\"" + self._relative_path +"/"+ self._lib_name +'.h'+"\""

    def lib_abs_path(self):
        return os.path.join(self._project_root_abs_path, self._relative_path)

    def lib_name(self):
        return self._lib_name

