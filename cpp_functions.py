from t1005_graph import *
from typing import List
from cpp_library import UserLibrary
from header import Header
import os

# DONE test , fuck you CONTINUE
class CppFunction(FunctionBase):
    """
    TODO(huaiyuan): support more than one function in a library
    """
    def __init__(self, library: UserLibrary):
        path = library.lib_abs_path()
        name = library.lib_name()
        header_file = os.path.join(path, name + ".h")

        assert os.path.exists(header_file), "header %s doesn't exist."%header_file

        lines = []
        with open(header_file, "r") as fp:
            lines = fp.readlines()

        # get the namespace
        self.calling_prefix = ""
        for i in range(len(lines)):
            id = lines[i].find("namespace")
            if id >= 0:
                left_bra = lines[i].find("{")
                if left_bra > id+9:
                    self.calling_prefix = lines[i][id+9:left_bra]+"::"
                    break

        header_begin = None
        header_end = None
        for i in range(len(lines)):
            if lines[i].find(library.header_begin_comments_h()) >= 0:
                header_begin = i
                break
        for i in range(len(lines)):
            if lines[i].find(library.header_end_comments_h()) >= 0:
                header_end = i
                break
        assert header_begin is not None, "Invalid header"
        assert header_end is not None, "Invalid header"
        assert header_begin < header_end, "Invalid header"

        self.header = Header.create_from_header_core(lines[header_begin+1:header_end])

        super(CppFunction, self).__init__(self.header.input_spec.copy(),
                                          self.header.output_spec.copy(),
                                          {library, },
                                          self.header.supported_options.copy())

    def optional_header(self) -> "Header":
        return self.header

    def print_call(self, full_context: FullContext) -> CallResult:
        return self.header.print_call(full_context,prefix=self.calling_prefix)
        # CONTINUE
