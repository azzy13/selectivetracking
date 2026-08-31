# msgs

Interface package for `groundingdino_ros`. Defines the detection and
perception message types the node publishes.

| Type              | Published on                        |
|-------------------|-------------------------------------|
| `DetectionArray`  | `/perception/detections`            |
| `PerceptionArray` | `/perception/perceptions`           |
| `PerceptionArray` | `/vanderbilt/fake_perception/data`  |

## Wire compatibility

ROS 2 matches a publisher to a subscriber by the fully-qualified type name
**and** the message definition. These types are `msgs/msg/...`, so a
subscriber expecting any other package name will not connect — silently, with
no error and no data. Changing a field here has the same effect.

Both sides of a topic must therefore be built from this package. If you need
to interoperate with a consumer built against a differently-named package,
that consumer has to be rebuilt against these definitions, or a translation
node has to sit at the boundary.

## Editing

Adding a field is safe for this repo's own nodes only after every publisher
and subscriber is rebuilt together. `Perception.frame_number` does not exist
here; `groundingdino_node.py` writes optional fields through a `hasattr` guard
(`_set_if_present`) so a build with extra fields still works.
