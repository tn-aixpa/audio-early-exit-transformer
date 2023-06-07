# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/conda/bin/cmake

# The command to remove a file.
RM = /opt/conda/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /falavi/slu/sentencepiece

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /falavi/slu/sentencepiece/build

# Include any dependencies generated for this target.
include src/CMakeFiles/spm_normalize.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/spm_normalize.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/spm_normalize.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/spm_normalize.dir/flags.make

src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o: src/CMakeFiles/spm_normalize.dir/flags.make
src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o: ../src/spm_normalize_main.cc
src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o: src/CMakeFiles/spm_normalize.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/falavi/slu/sentencepiece/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o"
	cd /falavi/slu/sentencepiece/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o -MF CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o.d -o CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o -c /falavi/slu/sentencepiece/src/spm_normalize_main.cc

src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.i"
	cd /falavi/slu/sentencepiece/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /falavi/slu/sentencepiece/src/spm_normalize_main.cc > CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.i

src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.s"
	cd /falavi/slu/sentencepiece/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /falavi/slu/sentencepiece/src/spm_normalize_main.cc -o CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.s

# Object files for target spm_normalize
spm_normalize_OBJECTS = \
"CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o"

# External object files for target spm_normalize
spm_normalize_EXTERNAL_OBJECTS =

src/spm_normalize: src/CMakeFiles/spm_normalize.dir/spm_normalize_main.cc.o
src/spm_normalize: src/CMakeFiles/spm_normalize.dir/build.make
src/spm_normalize: src/libsentencepiece_train.so.0.0.0
src/spm_normalize: src/libsentencepiece.so.0.0.0
src/spm_normalize: src/CMakeFiles/spm_normalize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/falavi/slu/sentencepiece/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable spm_normalize"
	cd /falavi/slu/sentencepiece/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spm_normalize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/spm_normalize.dir/build: src/spm_normalize
.PHONY : src/CMakeFiles/spm_normalize.dir/build

src/CMakeFiles/spm_normalize.dir/clean:
	cd /falavi/slu/sentencepiece/build/src && $(CMAKE_COMMAND) -P CMakeFiles/spm_normalize.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/spm_normalize.dir/clean

src/CMakeFiles/spm_normalize.dir/depend:
	cd /falavi/slu/sentencepiece/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /falavi/slu/sentencepiece /falavi/slu/sentencepiece/src /falavi/slu/sentencepiece/build /falavi/slu/sentencepiece/build/src /falavi/slu/sentencepiece/build/src/CMakeFiles/spm_normalize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/spm_normalize.dir/depend
