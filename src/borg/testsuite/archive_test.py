import json
import os
import errno
import stat
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from . import rejected_dotdot_paths
from ..crypto.key import PlaintextKey
from ..archive import Archive, CacheChunkBuffer, RobustUnpacker, valid_msgpacked_dict, ITEM_KEYS, Statistics
from ..archive import BackupOSError, backup_io, backup_io_iter, get_item_uid_gid, MetadataCollector, is_special
from ..archive import BackupIO, BackupPermissionError, BackupFileNotFoundError, stat_update_check, ChunksProcessor
from ..archive import BackupRaceConditionError, ChunkListEntry
from ..chunker import Chunk
from ..helpers import msgpack
from ..item import Item, ArchiveItem
from ..manifest import Manifest
from ..platform import uid2user, gid2group, is_win32


@pytest.fixture()
def stats():
    stats = Statistics()
    stats.update(20, unique=True)
    stats.nfiles = 1
    return stats


def test_stats_basic(stats):
    assert stats.osize == 20
    assert stats.usize == 20
    stats.update(20, unique=False)
    assert stats.osize == 40
    assert stats.usize == 20


@pytest.mark.parametrize(
    "item_path, update_size, expected_output",
    [
        ("", 0, "20 B O 20 B U 1 N "),  # test unchanged 'stats' fixture
        ("foo", 10**3, "1.02 kB O 20 B U 1 N foo"),  # test updated original size and set item path
        # test long item path which exceeds 80 characters
        ("foo" * 40, 10**3, "1.02 kB O 20 B U 1 N foofoofoofoofoofoofoofoofo...foofoofoofoofoofoofoofoofoofoo"),
    ],
)
def test_stats_progress(item_path, update_size, expected_output, stats, monkeypatch, columns=80):
    monkeypatch.setenv("COLUMNS", str(columns))
    out = StringIO()
    item = Item(path=item_path) if item_path else None
    s = expected_output

    stats.update(update_size, unique=False)
    stats.show_progress(item=item, stream=out)
    buf = " " * (columns - len(s))
    assert out.getvalue() == s + buf + "\r"


def test_stats_format(stats):
    assert (
        str(stats)
        == """\
Number of files: 1
Original size: 20 B
Deduplicated size: 20 B
Time spent in hashing: 0.000 seconds
Time spent in chunking: 0.000 seconds
Added files: 0
Unchanged files: 0
Modified files: 0
Error files: 0
Files changed while reading: 0
Bytes read from remote: 0
Bytes sent to remote: 0
"""
    )
    s = f"{stats.osize_fmt}"
    assert s == "20 B"
    # kind of redundant, but id is variable so we can't match reliably
    assert repr(stats) == f"<Statistics object at {id(stats):#x} (20, 20)>"


def test_stats_progress_json(stats):
    stats.output_json = True

    out = StringIO()
    stats.show_progress(item=Item(path="foo"), stream=out)
    result = json.loads(out.getvalue())
    assert result["type"] == "archive_progress"
    assert isinstance(result["time"], float)
    assert result["finished"] is False
    assert result["path"] == "foo"
    assert result["original_size"] == 20
    assert result["nfiles"] == 1

    out = StringIO()
    stats.show_progress(stream=out, final=True)
    result = json.loads(out.getvalue())
    assert result["type"] == "archive_progress"
    assert isinstance(result["time"], float)
    assert result["finished"] is True  # see #6570
    assert "path" not in result
    assert "original_size" not in result
    assert "nfiles" not in result


@pytest.mark.parametrize(
    "isoformat, expected",
    [
        ("1970-01-01T00:00:01.000001", datetime(1970, 1, 1, 0, 0, 1, 1, timezone.utc)),  # test with microseconds
        ("1970-01-01T00:00:01", datetime(1970, 1, 1, 0, 0, 1, 0, timezone.utc)),  # test without microseconds
    ],
)
def test_timestamp_parsing(monkeypatch, isoformat, expected):
    repository = Mock()
    key = PlaintextKey(repository)
    manifest = Manifest(key, repository)
    a = Archive(manifest, "test", create=True)
    a.metadata = ArchiveItem(time=isoformat)
    assert a.ts == expected


class MockCache:
    class MockRepo:
        def async_response(self, wait=True):
            pass

    def __init__(self):
        self.objects = {}
        self.repository = self.MockRepo()

    def add_chunk(self, id, meta, data, stats=None, wait=True, ro_type=None):
        assert ro_type is not None
        self.objects[id] = data
        return id, len(data)


def test_cache_chunk_buffer():
    data = [Item(path="p1"), Item(path="p2")]
    cache = MockCache()
    key = PlaintextKey(None)
    chunks = CacheChunkBuffer(cache, key, None)
    for d in data:
        chunks.add(d)
        chunks.flush()
    chunks.flush(flush=True)
    assert len(chunks.chunks) == 2
    unpacker = msgpack.Unpacker()
    for id in chunks.chunks:
        unpacker.feed(cache.objects[id])
    assert data == [Item(internal_dict=d) for d in unpacker]


def test_partial_cache_chunk_buffer():
    big = "0123456789abcdefghijklmnopqrstuvwxyz" * 25000
    data = [Item(path="full", target=big), Item(path="partial", target=big)]
    cache = MockCache()
    key = PlaintextKey(None)
    chunks = CacheChunkBuffer(cache, key, None)
    for d in data:
        chunks.add(d)
    chunks.flush(flush=False)
    # the code is expected to leave the last partial chunk in the buffer
    assert len(chunks.chunks) == 3
    assert chunks.buffer.tell() > 0
    # now really flush
    chunks.flush(flush=True)
    assert len(chunks.chunks) == 4
    assert chunks.buffer.tell() == 0
    unpacker = msgpack.Unpacker()
    for id in chunks.chunks:
        unpacker.feed(cache.objects[id])
    assert data == [Item(internal_dict=d) for d in unpacker]


def make_chunks(items):
    return b"".join(msgpack.packb({"path": item}) for item in items)


def _validator(value):
    return isinstance(value, dict) and value.get("path") in ("foo", "bar", "boo", "baz")


def process(input):
    unpacker = RobustUnpacker(validator=_validator, item_keys=ITEM_KEYS)
    result = []
    for should_sync, chunks in input:
        if should_sync:
            unpacker.resync()
        for data in chunks:
            unpacker.feed(data)
            for item in unpacker:
                result.append(item)
    return result


def test_extra_garbage_no_sync():
    chunks = [(False, [make_chunks(["foo", "bar"])]), (False, [b"garbage"] + [make_chunks(["boo", "baz"])])]
    res = process(chunks)
    assert res == [{"path": "foo"}, {"path": "bar"}, 103, 97, 114, 98, 97, 103, 101, {"path": "boo"}, {"path": "baz"}]


def split(left, length):
    parts = []
    while left:
        parts.append(left[:length])
        left = left[length:]
    return parts


def test_correct_stream():
    chunks = split(make_chunks(["foo", "bar", "boo", "baz"]), 2)
    input = [(False, chunks)]
    result = process(input)
    assert result == [{"path": "foo"}, {"path": "bar"}, {"path": "boo"}, {"path": "baz"}]


def test_missing_chunk():
    chunks = split(make_chunks(["foo", "bar", "boo", "baz"]), 4)
    input = [(False, chunks[:3]), (True, chunks[4:])]
    result = process(input)
    assert result == [{"path": "foo"}, {"path": "boo"}, {"path": "baz"}]


def test_corrupt_chunk():
    chunks = split(make_chunks(["foo", "bar", "boo", "baz"]), 4)
    input = [(False, chunks[:3]), (True, [b"gar", b"bage"] + chunks[3:])]
    result = process(input)
    assert result == [{"path": "foo"}, {"path": "boo"}, {"path": "baz"}]


@pytest.fixture
def item_keys_serialized():
    return [msgpack.packb(name) for name in ITEM_KEYS]


@pytest.mark.parametrize(
    "packed",
    [b"", b"x", b"foobar"]
    + [
        msgpack.packb(o)
        for o in (
            [None, 0, 0.0, False, "", {}, [], ()]
            + [42, 23.42, True, b"foobar", {b"foo": b"bar"}, [b"foo", b"bar"], (b"foo", b"bar")]
        )
    ],
)
def test_invalid_msgpacked_item(packed, item_keys_serialized):
    assert not valid_msgpacked_dict(packed, item_keys_serialized)


# pytest-xdist requires always same order for the keys and dicts:
IK = sorted(list(ITEM_KEYS))


@pytest.mark.parametrize(
    "packed",
    [
        msgpack.packb(o)
        for o in [
            {"path": b"/a/b/c"},  # small (different msgpack mapping type!)
            OrderedDict((k, b"") for k in IK),  # as big (key count) as it gets
            OrderedDict((k, b"x" * 1000) for k in IK),  # as big (key count and volume) as it gets
        ]
    ],
    ids=["minimal", "empty-values", "long-values"],
)
def test_valid_msgpacked_items(packed, item_keys_serialized):
    assert valid_msgpacked_dict(packed, item_keys_serialized)


def test_key_length_msgpacked_items():
    key = "x" * 32  # 31 bytes is the limit for fixstr msgpack type
    data = {key: b""}
    item_keys_serialized = [msgpack.packb(key)]
    assert valid_msgpacked_dict(msgpack.packb(data), item_keys_serialized)


def test_backup_io():
    with pytest.raises(BackupOSError):
        with backup_io:
            raise OSError(123)


def test_backup_io_iter():
    class Iterator:
        def __init__(self, exc):
            self.exc = exc

        def __next__(self):
            raise self.exc()

    oserror_iterator = Iterator(OSError)
    with pytest.raises(BackupOSError):
        for _ in backup_io_iter(oserror_iterator):
            pass

    normal_iterator = Iterator(StopIteration)
    for _ in backup_io_iter(normal_iterator):
        assert False, "StopIteration handled incorrectly"


def test_get_item_uid_gid():
    # test requires that:
    # - a user/group name for the current process' real uid/gid exists.
    # - a system user/group udoesnotexist:gdoesnotexist does NOT exist.

    try:
        puid, pgid = os.getuid(), os.getgid()  # UNIX only
    except AttributeError:
        puid, pgid = 0, 0
    puser, pgroup = uid2user(puid), gid2group(pgid)

    # this is intentionally a "strange" item, with not matching ids/names.
    item = Item(path="filename", uid=1, gid=2, user=puser, group=pgroup)

    uid, gid = get_item_uid_gid(item, numeric=False)
    # these are found via a name-to-id lookup
    assert uid == puid
    assert gid == pgid

    uid, gid = get_item_uid_gid(item, numeric=True)
    # these are directly taken from the item.uid and .gid
    assert uid == 1
    assert gid == 2

    uid, gid = get_item_uid_gid(item, numeric=False, uid_forced=3, gid_forced=4)
    # these are enforced (not from item metadata)
    assert uid == 3
    assert gid == 4

    # item metadata broken, has negative ids.
    item = Item(path="filename", uid=-1, gid=-2, user=puser, group=pgroup)

    uid, gid = get_item_uid_gid(item, numeric=True)
    # use the uid/gid defaults (which both default to 0).
    assert uid == 0
    assert gid == 0

    uid, gid = get_item_uid_gid(item, numeric=True, uid_default=5, gid_default=6)
    # use the uid/gid defaults (as given).
    assert uid == 5
    assert gid == 6

    # item metadata broken, has negative ids and non-existing user/group names.
    item = Item(path="filename", uid=-3, gid=-4, user="udoesnotexist", group="gdoesnotexist")

    uid, gid = get_item_uid_gid(item, numeric=False)
    # use the uid/gid defaults (which both default to 0).
    assert uid == 0
    assert gid == 0

    uid, gid = get_item_uid_gid(item, numeric=True, uid_default=7, gid_default=8)
    # use the uid/gid defaults (as given).
    assert uid == 7
    assert gid == 8

    if not is_win32:
        # due to the hack in borg.platform.windows user2uid / group2gid, these always return 0
        # (no matter which username we ask for) and they never raise a KeyError (like e.g. for
        # a non-existing user/group name). Thus, these tests can currently not succeed on win32.

        # item metadata has valid uid/gid, but non-existing user/group names.
        item = Item(path="filename", uid=9, gid=10, user="udoesnotexist", group="gdoesnotexist")

        uid, gid = get_item_uid_gid(item, numeric=False)
        # because user/group name does not exist here, use valid numeric ids from item metadata.
        assert uid == 9
        assert gid == 10

        uid, gid = get_item_uid_gid(item, numeric=False, uid_default=11, gid_default=12)
        # because item uid/gid seems valid, do not use the given uid/gid defaults
        assert uid == 9
        assert gid == 10

    # item metadata only has uid/gid, but no user/group.
    item = Item(path="filename", uid=13, gid=14)

    uid, gid = get_item_uid_gid(item, numeric=False)
    # it'll check user/group first, but as there is nothing in the item, falls back to uid/gid.
    assert uid == 13
    assert gid == 14

    uid, gid = get_item_uid_gid(item, numeric=True)
    # does not check user/group, directly returns uid/gid.
    assert uid == 13
    assert gid == 14

    # item metadata has no uid/gid/user/group.
    item = Item(path="filename")

    uid, gid = get_item_uid_gid(item, numeric=False, uid_default=15)
    # as there is nothing, it'll fall back to uid_default/gid_default.
    assert uid == 15
    assert gid == 0

    uid, gid = get_item_uid_gid(item, numeric=True, gid_default=16)
    # as there is nothing, it'll fall back to uid_default/gid_default.
    assert uid == 0
    assert gid == 16


def test_reject_non_sanitized_item():
    for path in rejected_dotdot_paths:
        with pytest.raises(ValueError, match="unexpected '..' element in path"):
            Item(path=path, user="root", group="root")


def test_is_special():
    """Test the is_special function that identifies special files."""
    # Regular file
    assert not is_special(stat.S_IFREG | 0o644)
    # Directory
    assert not is_special(stat.S_IFDIR | 0o755)
    # Symlink
    assert not is_special(stat.S_IFLNK | 0o0)
    # FIFO (special)
    assert is_special(stat.S_IFIFO | 0o644)
    # Character device (special)
    assert is_special(stat.S_IFCHR | 0o644)
    # Block device (special)
    assert is_special(stat.S_IFBLK | 0o644)
    # Socket (not considered special by is_special)
    assert not is_special(stat.S_IFSOCK | 0o644)


def test_archive_methods():
    """Test various methods of the Archive class."""
    repository = Mock()
    key = PlaintextKey(repository)
    manifest = Manifest(key, repository)

    # Create a test archive
    archive = Archive(manifest, "test-archive", create=True)

    # Set metadata for testing
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_time = start_time + timedelta(minutes=5)

    # Set start and end attributes directly
    archive.start = start_time
    archive.end = end_time

    # Set id attribute for fpr property test
    archive.id = b"01234567012345670123456701234567"  # 256bit

    archive.metadata = ArchiveItem(
        time=start_time.isoformat(),
        time_end=end_time.isoformat(),
        command_line="borg create test-archive",  # Command line must be a string, not a list
        hostname="test-host",
        username="test-user",
        comment="Test archive comment",
        chunker_params=(1, 2, 3, 4),
    )

    # Test ts property
    assert archive.ts == start_time

    # Test ts_end property
    assert archive.ts_end == end_time

    # Test duration property (returns a formatted string)
    assert archive.duration == "5 minutes 0.000 seconds"

    # Test duration_from_meta property (returns a formatted string)
    assert archive.duration_from_meta == "5 minutes 0.000 seconds"

    # Test fpr property - it returns a hex representation of the ID
    assert archive.fpr == "3031323334353637303132333435363730313233343536373031323334353637"

    # Test info() method
    info = archive.info()

    # When create=True, the info() method returns a different set of fields
    # than when create=False. Let's check the fields that should be present.
    assert info["name"] == "test-archive"
    assert info["id"] == "3031323334353637303132333435363730313233343536373031323334353637"
    assert info["duration"] == 300.0  # 5 minutes in seconds
    assert "start" in info
    assert "end" in info
    assert "stats" in info
    assert "command_line" in info


def test_metadata_collector_stat_simple_attrs():
    """Test the MetadataCollector's stat_simple_attrs method."""
    # Setup mock stat result with concrete values
    mock_stat_result = Mock()
    mock_stat_result.st_mode = stat.S_IFREG | 0o644  # Regular file
    mock_stat_result.st_uid = 1000
    mock_stat_result.st_gid = 1000
    mock_stat_result.st_mtime_ns = 1609459200000000000  # 2021-01-01 00:00:00
    mock_stat_result.st_atime_ns = 1609459200000000000
    mock_stat_result.st_ctime_ns = 1609459200000000000
    mock_stat_result.st_ino = 12345

    # Mock uid2user and gid2group functions
    with (
        patch("borg.archive.uid2user", return_value="testuser"),
        patch("borg.archive.gid2group", return_value="testgroup"),
        patch("borg.archive.get_birthtime_ns", return_value=None),
    ):

        # Create a MetadataCollector instance
        collector = MetadataCollector(
            noatime=False, noctime=False, nobirthtime=False, numeric_ids=False, noflags=True, noacls=True, noxattrs=True
        )

        # Call the method under test
        result = collector.stat_simple_attrs(mock_stat_result, "/test/path")

        # Verify the result
        assert result["mode"] == stat.S_IFREG | 0o644
        assert result["uid"] == 1000
        assert result["gid"] == 1000
        assert result["user"] == "testuser"
        assert result["group"] == "testgroup"
        assert result["mtime"] == 1609459200000000000
        assert result["atime"] == 1609459200000000000
        assert result["ctime"] == 1609459200000000000
        assert result["inode"] == 12345


def test_metadata_collector_stat_ext_attrs():
    """Test the MetadataCollector's stat_ext_attrs method."""
    # Setup mock stat result
    mock_stat_result = Mock()
    mock_stat_result.st_mode = stat.S_IFREG | 0o644  # Regular file

    # Test with xattrs enabled
    # Mock acl_get to add acl_access to the item dictionary
    def mock_acl_get(path, item, st, numeric_ids=False, fd=None):
        item["acl_access"] = b"mock_acl_data"
        return True

    with (
        patch("borg.xattr.get_all", return_value={b"test_xattr": b"test_value"}),
        patch("borg.archive.get_flags", return_value=42),
        patch("borg.archive.acl_get", side_effect=mock_acl_get),
    ):

        # Create a MetadataCollector instance with xattrs enabled
        collector = MetadataCollector(
            noatime=True,
            noctime=True,
            nobirthtime=True,
            numeric_ids=True,
            noflags=False,  # Enable flags
            noacls=False,  # Enable ACLs
            noxattrs=False,  # Enable xattrs
        )

        # Call the method under test
        result = collector.stat_ext_attrs(mock_stat_result, "/test/path")

        # Verify the result - should include xattrs, bsdflags, and acl_access
        assert result["xattrs"] == {b"test_xattr": b"test_value"}
        assert result["bsdflags"] == 42
        assert result["acl_access"] == b"mock_acl_data"

    # Test with all extended attributes disabled
    with patch("borg.xattr.get_all") as mock_xattr_get_all:
        # Create a MetadataCollector instance with all ext attrs disabled
        collector = MetadataCollector(
            noatime=True,
            noctime=True,
            nobirthtime=True,
            numeric_ids=True,
            noflags=True,  # Disable flags
            noacls=True,  # Disable ACLs
            noxattrs=True,  # Disable xattrs
        )

        # Call the method under test
        result = collector.stat_ext_attrs(mock_stat_result, "/test/path")

        # Verify the result - should be empty
        assert result == {}

        # xattr.get_all should not be called when noxattrs=True
        mock_xattr_get_all.assert_not_called()


def test_archive_rename():
    """Test the rename method of the Archive class."""
    repository = Mock()
    key = PlaintextKey(repository)
    manifest = Manifest(key, repository)

    # Mock the archives dictionary and methods
    manifest.archives = Mock()
    manifest.archives.delete_by_id = Mock()
    manifest.archives.create = Mock()

    # Create a test archive
    archive = Archive(manifest, "old-name", create=True)
    archive.id = b"old-id"  # Set an ID manually

    # Mock the set_meta method to avoid actual metadata updates
    archive.set_meta = Mock()

    # Rename the archive
    archive.rename("new-name")

    # Check that the name was updated
    assert archive.name == "new-name"

    # Check that set_meta was called with the new name
    archive.set_meta.assert_called_once_with("name", "new-name")

    # Check that the old archive was deleted from the manifest
    manifest.archives.delete_by_id.assert_called_once_with(b"old-id")


def test_archive_delete():
    """Test the delete method of the Archive class."""
    repository = Mock()
    key = PlaintextKey(repository)
    manifest = Manifest(key, repository)

    # Mock the archives dictionary and methods
    manifest.archives = Mock()
    manifest.archives.delete_by_id = Mock()

    # Create a test archive
    archive = Archive(manifest, "test-archive", create=True)
    archive.id = b"test-id"  # Set an ID manually

    # Delete the archive
    archive.delete()

    # Check that the archive was removed from the manifest
    manifest.archives.delete_by_id.assert_called_once_with(b"test-id")


def test_chunks_processor():
    """Test the ChunksProcessor class."""
    # Create mocks
    key = Mock()
    cache = Mock()
    add_item = Mock()

    # Create a ChunksProcessor instance
    processor = ChunksProcessor(key=key, cache=cache, add_item=add_item, rechunkify=False)

    # Create test data
    item = Item(path="test-file", mode=stat.S_IFREG | 0o644)
    stats = Statistics()

    # Mock the chunk iterator
    chunk1 = Chunk(id=None, data=b"chunk1data", size=len(b"chunk1data"))
    chunk2 = Chunk(id=None, data=b"chunk2data", size=len(b"chunk2data"))
    chunk_iter = [chunk1, chunk2]

    # Mock cached_hash to return chunk_id and data
    with patch("borg.archive.cached_hash") as mock_cached_hash:
        mock_cached_hash.side_effect = [(b"chunk1id", b"chunk1data"), (b"chunk2id", b"chunk2data")]

        # Mock cache.add_chunk to return chunk entries
        # The format is (chunk_id, size, csize) based on the assertion below
        cache.add_chunk.side_effect = [
            (b"chunk1id", len(b"chunk1data"), len(b"chunk1data")),
            (b"chunk2id", len(b"chunk2data"), len(b"chunk2data")),
        ]

        # Mock repository.async_response
        cache.repository = Mock()
        cache.repository.async_response = Mock()

        # Call the method under test
        processor.process_file_chunks(item=item, cache=cache, stats=stats, show_progress=False, chunk_iter=chunk_iter)

    # Verify that cache.add_chunk was called for each chunk
    assert cache.add_chunk.call_count == 2

    # Verify that the item's chunks were updated
    assert item.get("chunks") == [
        (b"chunk1id", len(b"chunk1data"), len(b"chunk1data")),
        (b"chunk2id", len(b"chunk2data"), len(b"chunk2data")),
    ]


def test_archive_compare_archives_iter():
    """Test the compare_archives_iter method of the Archive class."""
    # Create mock repositories and keys
    repository1 = Mock()
    repository2 = Mock()
    key1 = PlaintextKey(repository1)
    key2 = PlaintextKey(repository2)
    manifest1 = Manifest(key1, repository1)
    manifest2 = Manifest(key2, repository2)

    # Create test archives
    archive1 = Archive(manifest1, "archive1", create=True)
    archive2 = Archive(manifest2, "archive2", create=True)

    # Mock the pipeline attribute for both archives
    archive1.pipeline = Mock()
    archive2.pipeline = Mock()

    # Mock fetch_many to return appropriate chunk data for comparison
    # We need to return iterators that yield the chunk data
    def fetch_many_side_effect(chunks, ro_type=None):
        chunk_id = chunks[0].id if chunks else None
        if chunk_id == b"chunk1":
            return iter([b"same content"])
        elif chunk_id == b"chunk2":
            return iter([b"content only in archive1"])
        elif chunk_id == b"chunk3":
            return iter([b"content only in archive2"])
        elif chunk_id == b"chunk4":
            return iter([b"different content 1"])
        elif chunk_id == b"chunk5":
            return iter([b"different content 2"])
        return iter([])

    archive1.pipeline.fetch_many = Mock(side_effect=fetch_many_side_effect)
    archive2.pipeline.fetch_many = Mock(side_effect=fetch_many_side_effect)

    # Create a mock matcher
    class MockMatcher:
        def match(self, path):
            return True

    matcher = MockMatcher()

    # Mock iter_items to return test items
    # Create separate but identical items for each archive
    # Use ChunkListEntry objects instead of tuples for the chunks
    common_item1 = Item(path="common", mode=stat.S_IFREG | 0o644, size=100, chunks=[ChunkListEntry(b"chunk1", 100)])
    common_item2 = Item(path="common", mode=stat.S_IFREG | 0o644, size=100, chunks=[ChunkListEntry(b"chunk1", 100)])
    only_in_1 = Item(path="only_in_1", mode=stat.S_IFREG | 0o644, size=200, chunks=[ChunkListEntry(b"chunk2", 200)])
    only_in_2 = Item(path="only_in_2", mode=stat.S_IFREG | 0o644, size=300, chunks=[ChunkListEntry(b"chunk3", 300)])
    different_item1 = Item(
        path="different", mode=stat.S_IFREG | 0o644, size=400, chunks=[ChunkListEntry(b"chunk4", 400)]
    )
    different_item2 = Item(
        path="different", mode=stat.S_IFREG | 0o644, size=500, chunks=[ChunkListEntry(b"chunk5", 500)]
    )  # Different size

    archive1.iter_items = Mock(return_value=[common_item1, only_in_1, different_item1])
    archive2.iter_items = Mock(return_value=[common_item2, only_in_2, different_item2])

    # Call the method under test
    # Set can_compare_chunk_ids=True to allow comparing chunk IDs
    result = list(Archive.compare_archives_iter(archive1, archive2, matcher=matcher, can_compare_chunk_ids=True))

    # Verify the result - we expect 4 items: common, different, only_in_1, only_in_2
    assert len(result) == 4

    # Check that we have the expected paths
    paths = [item.path for item in result]
    assert "common" in paths
    assert "different" in paths
    assert "only_in_1" in paths
    assert "only_in_2" in paths

    # Check that the ItemDiff objects have the expected changes
    for item in result:
        if item.path == "common":
            # When can_compare_chunk_ids=True, it compares chunk IDs which are the same
            # But the content comparison still happens and shows differences
            # So we check that there are changes but they're only in content
            changes = item.changes()
            assert len(changes) == 1
            assert "content" in changes
        elif item.path == "different":
            # Different item should have content changes
            assert "content" in item.changes()
        elif item.path == "only_in_1":
            # Item only in archive1 should be marked as removed
            assert "content" in item.changes()
            assert item.changes()["content"].diff_type == "removed"
        elif item.path == "only_in_2":
            # Item only in archive2 should be marked as added
            assert "content" in item.changes()
            assert item.changes()["content"].diff_type == "added"


def test_archive_set_meta():
    """Test the set_meta method of the Archive class."""
    repository = Mock()
    key = PlaintextKey(repository)
    manifest = Manifest(key, repository)

    # Create a test archive
    archive = Archive(manifest, "test-archive", create=True)
    archive.id = b"test-id"  # Set an ID manually

    # Create a metadata item with a time attribute
    metadata = ArchiveItem(time="2023-01-01T00:00:00")

    # Mock the necessary methods and attributes
    archive._load_meta = Mock(return_value=metadata)
    archive.key = Mock()
    archive.key.pack_metadata = Mock(return_value=b"packed-metadata")
    archive.key.id_hash = Mock(return_value=b"new-id")
    archive.cache = Mock()
    archive.cache.add_chunk = Mock()
    archive.stats = None
    manifest.archives.create = Mock()

    # Set metadata
    archive.set_meta("comment", "New comment")

    # Verify the methods were called correctly
    archive._load_meta.assert_called_once_with(b"test-id")
    archive.key.pack_metadata.assert_called_once()
    archive.key.id_hash.assert_called_once_with(b"packed-metadata")
    archive.cache.add_chunk.assert_called_once()
    manifest.archives.create.assert_called_once_with("test-archive", b"new-id", "2023-01-01T00:00:00", overwrite=True)

    # Verify the ID was updated
    assert archive.id == b"new-id"

    # Verify the metadata was updated
    assert archive._load_meta.return_value.comment == "New comment"


def test_backup_io_class():
    """Test the BackupIO context manager class."""
    # Test normal operation (no exception)
    bio = BackupIO()
    # Call the object to test the __call__ method
    assert bio("test operation") is bio

    # Test context manager with no exception
    with BackupIO():
        pass  # No exception should be raised

    # Test with OSError
    with pytest.raises(BackupOSError):
        with BackupIO():
            raise OSError(123, "Test error")

    # Test with specific OSError subclasses
    with pytest.raises(BackupPermissionError):
        with BackupIO():
            raise OSError(errno.EACCES, "Permission denied")

    with pytest.raises(BackupFileNotFoundError):
        with BackupIO():
            raise OSError(errno.ENOENT, "File not found")

    # Test with non-OSError exception
    with pytest.raises(ValueError):
        with BackupIO():
            raise ValueError("Not an OSError")


def test_stat_update_check():
    """Test the stat_update_check function."""
    # Create a namedtuple for stat objects to ensure they have the right attributes
    from collections import namedtuple

    StatResult = namedtuple("StatResult", ["st_mode", "st_mtime_ns", "st_ctime_ns", "st_size", "st_ino", "st_dev"])

    # Create base stat object
    st_old = StatResult(
        st_mode=stat.S_IFREG | 0o644,  # Regular file
        st_mtime_ns=1000000000,
        st_ctime_ns=1000000000,
        st_size=100,
        st_ino=123,
        st_dev=456,
    )

    # Test with identical stats (no change)
    st_curr = st_old

    # Function should return the current stat result for identical stats
    assert stat_update_check(st_old, st_curr) == st_curr

    # Test with changed mtime
    st_curr = StatResult(
        st_mode=stat.S_IFREG | 0o644,  # Regular file
        st_mtime_ns=2000000000,  # Changed
        st_ctime_ns=1000000000,
        st_size=100,
        st_ino=123,
        st_dev=456,
    )

    assert stat_update_check(st_old, st_curr) == st_curr

    # Test with changed ctime
    st_curr = StatResult(
        st_mode=stat.S_IFREG | 0o644,  # Regular file
        st_mtime_ns=1000000000,
        st_ctime_ns=2000000000,  # Changed
        st_size=100,
        st_ino=123,
        st_dev=456,
    )

    assert stat_update_check(st_old, st_curr) == st_curr

    # Test with changed size
    st_curr = StatResult(
        st_mode=stat.S_IFREG | 0o644,  # Regular file
        st_mtime_ns=1000000000,
        st_ctime_ns=1000000000,
        st_size=200,  # Changed
        st_ino=123,
        st_dev=456,
    )

    assert stat_update_check(st_old, st_curr) == st_curr

    # Test with changed device (file moved)
    st_curr = StatResult(
        st_mode=stat.S_IFREG | 0o644,  # Regular file
        st_mtime_ns=1000000000,
        st_ctime_ns=1000000000,
        st_size=100,
        st_ino=123,
        st_dev=999,  # Changed
    )

    # Function should return the current stat result for changed device
    assert stat_update_check(st_old, st_curr) == st_curr

    # Test with changed inode (file replaced)
    st_curr = StatResult(
        st_mode=stat.S_IFREG | 0o644,  # Regular file
        st_mtime_ns=1000000000,
        st_ctime_ns=1000000000,
        st_size=100,
        st_ino=999,  # Changed
        st_dev=456,
    )

    # Should raise BackupRaceConditionError for changed inode
    with pytest.raises(BackupRaceConditionError):
        stat_update_check(st_old, st_curr)

    # Test with changed file type
    st_curr = StatResult(
        st_mode=stat.S_IFDIR | 0o755,  # Directory (changed from regular file)
        st_mtime_ns=1000000000,
        st_ctime_ns=1000000000,
        st_size=100,
        st_ino=123,
        st_dev=456,
    )

    # Should raise BackupRaceConditionError for changed file type
    with pytest.raises(BackupRaceConditionError):
        stat_update_check(st_old, st_curr)
