# Build the minimal aclnnAdd demo (MSPTI injection validation target).
#
# CANN toolkit root defaults to /usr/local/Ascend/cann-9.1.0, overridable via
# ASCEND_HOME_PATH (the standard Ascend env var) or CANN directly, e.g.:
#   make CANN=/usr/local/Ascend/cann-8.5.2

CANN ?= $(or $(ASCEND_HOME_PATH),/usr/local/Ascend/cann-9.1.0)

CXX      ?= g++
CXXFLAGS ?= -std=c++14
CPPFLAGS += -I$(CANN)/include
LDFLAGS  += -L$(CANN)/lib64 -Wl,-rpath,$(CANN)/lib64
LDLIBS   += -lascendcl -lnnopbase -lopapi

TARGET := add_demo
SRCS   := add_demo.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
