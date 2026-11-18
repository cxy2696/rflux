# rflux developer build.  CMake is the supported install path; this Makefile is
# intentionally small and is used for local tests and examples.
#
#   make            shared library, static C++ core, examples
#   make check      C ABI, C++ core, Python, version and docs checks
#   make examples   build C and C++ example programs
#   make clean

CC       ?= cc
CXX      ?= c++
PYTHON   ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
OPT      ?= -O2 -g
CSTD     ?= -std=c11
CXXSTD   ?= -std=c++17
WARN     ?= -Wall -Wextra -Wpedantic -Werror
CPPFLAGS += -Iinclude -Isrc
CFLAGS   += $(CSTD) $(WARN) -fPIC -MMD -MP
CXXFLAGS += $(CXXSTD) $(WARN) -fPIC -fvisibility=hidden -MMD -MP $(THREAD_FLAGS)
LDLIBS   += -lm $(THREAD_FLAGS)
PUBLIC_HEADERS := $(wildcard include/rflux/*.h) $(wildcard include/rflux/*/*.hpp)

BUILD    := build
CORE_SRCS := $(wildcard src/qflux/*.cpp) $(wildcard src/regimelab/*.cpp) \
             $(wildcard src/percolab/*.cpp) $(wildcard src/factorlab/*.cpp) \
             $(wildcard src/portfoliolab/*.cpp) $(wildcard src/execution/*.cpp) \
             $(wildcard src/mllab/*.cpp) $(wildcard src/risk/*.cpp)
CAPI_SRC  := src/rflux_capi.cpp
CORE_OBJS := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(CORE_SRCS))
CAPI_OBJ  := $(BUILD)/obj/src/rflux_capi.o
CORE_LIB  := $(BUILD)/librflux_core.a

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_LIB := $(BUILD)/librflux.dylib
SHARED_FLAGS := -dynamiclib
THREAD_FLAGS :=
else
SHARED_LIB := $(BUILD)/librflux.so
SHARED_FLAGS := -shared
# libstdc++ still needs the pthread flag on glibc versions older than 2.34.
THREAD_FLAGS := -pthread
endif
SHARED_LIB_ABS := $(abspath $(SHARED_LIB))
RPATH := -Wl,-rpath,$(abspath $(BUILD))

EXAMPLES := $(BUILD)/example_c_basic $(BUILD)/example_cpp_research
TESTS    := $(BUILD)/test_capi

.PHONY: all libs shared examples check c-test python-test python-lib algo-check version-check docs-check dup-check native wheel clean
all: libs examples

libs: $(SHARED_LIB) $(CORE_LIB)

$(BUILD)/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(OPT) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(OPT) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(CORE_LIB): $(CORE_OBJS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

shared: $(SHARED_LIB)

$(SHARED_LIB): $(CORE_OBJS) $(CAPI_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(SHARED_FLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_capi: tests/c/test_capi.c $(SHARED_LIB) $(PUBLIC_HEADERS)
	$(CC) $(OPT) $(CSTD) $(WARN) $(CPPFLAGS) -o $@ $< -L$(BUILD) -lrflux $(RPATH) $(LDLIBS)

$(BUILD)/example_c_basic: examples/c_basic.c $(SHARED_LIB) $(PUBLIC_HEADERS)
	$(CC) $(OPT) $(CSTD) $(WARN) $(CPPFLAGS) -o $@ $< -L$(BUILD) -lrflux $(RPATH) $(LDLIBS)

$(BUILD)/example_cpp_research: examples/cpp_research.cpp $(CORE_LIB) $(PUBLIC_HEADERS)
	$(CXX) $(OPT) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(CORE_LIB) $(LDLIBS)

examples: $(EXAMPLES)

c-test: $(BUILD)/test_capi
	$(BUILD)/test_capi

python-test: $(SHARED_LIB)
	RFLUX_LIBRARY=$(SHARED_LIB_ABS) PYTHONPATH=python $(PYTHON) -m unittest discover -s tests/python -p 'test_*.py' -v

algo-check: $(SHARED_LIB)
	RFLUX_LIBRARY=$(SHARED_LIB_ABS) PYTHONPATH=python $(PYTHON) scripts/verify_algorithms.py --label native
	RFLUX_DISABLE_NATIVE=1 RFLUX_LIBRARY=$(SHARED_LIB_ABS) PYTHONPATH=python $(PYTHON) scripts/verify_algorithms.py --label ctypes-fallback

python-lib: $(SHARED_LIB)
	@mkdir -p python/rflux/_lib
	cp $(SHARED_LIB) python/rflux/_lib/
	@echo "copied $(SHARED_LIB) -> python/rflux/_lib/"

native:
	$(PYTHON) -m pip install -e . --no-deps

wheel:
	$(PYTHON) -m pip wheel . --no-deps -w dist

version-check:
	$(PYTHON) scripts/check_version.py

docs-check:
	$(PYTHON) scripts/check_docs.py

# jscpd needs Node/npx, so CI does not run it; use it before large refactors.
dup-check:
	npx --yes jscpd@4 src include python/rflux tests \
		--min-lines 5 --min-tokens 40 --threshold 0 --reporters console

check: c-test python-test algo-check version-check docs-check

clean:
	rm -rf $(BUILD)

-include $(CORE_OBJS:.o=.d) $(CAPI_OBJ:.o=.d) \
         $(BUILD)/obj/tests/c/test_capi.d \
