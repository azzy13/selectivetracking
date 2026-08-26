# trinity_msgs (vendored)

Not our package. This is a **verbatim copy** of the SRI `trinity_msgs`
interface package at revision `31287a1` ("message definitions for perception
0.22") — the revision the running `architecture_demo` stack pins its
`trinity_msgs` submodule at.

## Why it is checked in

`groundingdino_ros` publishes `trinity_msgs/msg/PerceptionArray`. ROS 2 matches
a publisher to a subscriber by the fully-qualified type name and the message
definition, so the type has to be generated from a package that really is named
`trinity_msgs` and really has these fields. Renaming the package or editing a
field would produce a topic the stack never connects to — **silently**, with no
error and no data.

Vendoring the six files makes `docker/build_ros2.sh` self-contained: it does not
need a `trinity_msgs` checkout next to this repo, which is machine-specific and
is not something a clone of this repo has.

## Do not edit these files

Treat this directory as read-only. The field definitions are a wire contract
with the rest of the ANSR stack; changing them here does not change the stack,
it just stops the two from talking. Upstream changes come in by re-copying:

```bash
git -C /path/to/trinity_msgs archive <ref> | tar -x -C ros2_package/trinity_msgs
```

To build against a different revision **without** re-vendoring — e.g. to test
against 0.58 head — point the build script at a checkout:

```bash
TRINITY_MSGS_REF=master TRINITY_MSGS_SRC=/path/to/trinity_msgs ./docker/build_ros2.sh
```

See the "Message version skew" section of `DEMO.md` for how 0.22 and 0.58
differ and why the node writes version-dependent fields through a `hasattr`
guard.
