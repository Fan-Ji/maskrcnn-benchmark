# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer as google___protobuf___internal___containers___RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    Optional as typing___Optional,
    Text as typing___Text,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


class InferenceOptions(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    num_image_returned = ... # type: int

    def __init__(self,
        *,
        num_image_returned : typing___Optional[int] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> InferenceOptions: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"num_image_returned"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"num_image_returned",b"num_image_returned"]) -> None: ...

class Image(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    images_data = ... # type: bytes
    name = ... # type: typing___Text

    def __init__(self,
        *,
        images_data : typing___Optional[bytes] = None,
        name : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Image: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"images_data",u"name"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"images_data",b"images_data",u"name",b"name"]) -> None: ...

class ImageBatchRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def opt(self) -> InferenceOptions: ...

    @property
    def images(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Image]: ...

    def __init__(self,
        *,
        opt : typing___Optional[InferenceOptions] = None,
        images : typing___Optional[typing___Iterable[Image]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ImageBatchRequest: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"opt"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"images",u"opt"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"opt",b"opt"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"images",b"images",u"opt",b"opt"]) -> None: ...

class Rectangle(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    xlt = ... # type: float
    ylt = ... # type: float
    xrb = ... # type: float
    yrb = ... # type: float

    def __init__(self,
        *,
        xlt : typing___Optional[float] = None,
        ylt : typing___Optional[float] = None,
        xrb : typing___Optional[float] = None,
        yrb : typing___Optional[float] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Rectangle: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"xlt",u"xrb",u"ylt",u"yrb"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"xlt",b"xlt",u"xrb",b"xrb",u"ylt",b"ylt",u"yrb",b"yrb"]) -> None: ...

class RLE(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    size = ... # type: google___protobuf___internal___containers___RepeatedScalarFieldContainer[int]
    counts = ... # type: typing___Text

    def __init__(self,
        *,
        size : typing___Optional[typing___Iterable[int]] = None,
        counts : typing___Optional[typing___Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> RLE: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"counts",u"size"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"counts",b"counts",u"size",b"size"]) -> None: ...

class Detection(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    cropped = ... # type: bool
    category = ... # type: int
    confidence = ... # type: float

    @property
    def rle(self) -> RLE: ...

    @property
    def bbox(self) -> Rectangle: ...

    def __init__(self,
        *,
        rle : typing___Optional[RLE] = None,
        cropped : typing___Optional[bool] = None,
        category : typing___Optional[int] = None,
        confidence : typing___Optional[float] = None,
        bbox : typing___Optional[Rectangle] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Detection: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def HasField(self, field_name: typing_extensions___Literal[u"bbox",u"rle"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"bbox",u"category",u"confidence",u"cropped",u"rle"]) -> None: ...
    else:
        def HasField(self, field_name: typing_extensions___Literal[u"bbox",b"bbox",u"rle",b"rle"]) -> bool: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"bbox",b"bbox",u"category",b"category",u"confidence",b"confidence",u"cropped",b"cropped",u"rle",b"rle"]) -> None: ...

class ResultPerImage(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    image_id = ... # type: typing___Text

    @property
    def detections(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Detection]: ...

    def __init__(self,
        *,
        image_id : typing___Optional[typing___Text] = None,
        detections : typing___Optional[typing___Iterable[Detection]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ResultPerImage: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"detections",u"image_id"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"detections",b"detections",u"image_id",b"image_id"]) -> None: ...

class InferenceResult(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def returned_images(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[Image]: ...

    @property
    def result(self) -> google___protobuf___internal___containers___RepeatedCompositeFieldContainer[ResultPerImage]: ...

    def __init__(self,
        *,
        returned_images : typing___Optional[typing___Iterable[Image]] = None,
        result : typing___Optional[typing___Iterable[ResultPerImage]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> InferenceResult: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    if sys.version_info >= (3,):
        def ClearField(self, field_name: typing_extensions___Literal[u"result",u"returned_images"]) -> None: ...
    else:
        def ClearField(self, field_name: typing_extensions___Literal[u"result",b"result",u"returned_images",b"returned_images"]) -> None: ...
