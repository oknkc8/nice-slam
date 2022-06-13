# capture/sensor_data.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
from typing import *
from struct import unpack, pack
from enum import Enum

import numpy as np

from src.omni_utils.log import LOG_INFO, LOG_ERROR, LOG_WARNING

class SensorDataType(Enum):
    UNKNOWN = 0
    CAMERA_FRAME = 0x0100
    IMU = 0x0200
    LIDAR = 0x0300

class SensorDataHeader:
    def __init__(self):
        self.capture_desc = ''
        self.field_names = []

class SensorData:
    def __init__(self, sensor_id: int):
        self.sensor_id = sensor_id
        self.timestamp_ns = 0
        self.data = np.array([])

    def getHeader(self) -> SensorDataHeader:
        raise NotImplementedError
    def getTextData(self) -> List[str]:
        raise NotImplementedError
    def setMetaFromHeader(self, capture_desc: List[str]) -> bool:
        raise NotImplementedError
    def setMetaFromData(self, data: SensorData) -> bool:
        raise NotImplementedError
    def setFromTextData(self, text_data: List[str]) -> bool:
        raise NotImplementedError

    @staticmethod
    def getType(sensor_id: int) -> int:
        return sensor_id & 0xff00

    def type(self):
        return SensorData.getType(self.sensor_id)


class CameraFrameData(SensorData):
    DATA_TYPE = SensorDataType.CAMERA_FRAME
    class PixelFormat(Enum): UNKNOWN = 0; GRAY8 = 1; RGB8 = 2; GRAY16 = 3
    @staticmethod
    def pixelBytes(f: CameraFrameData.PixelFormat) -> int:
        if f == CameraFrameData.PixelFormat.RGB8: return 3
        elif f == CameraFrameData.PixelFormat.GRAY16: return 2
        else: return 1

    class Meta:
        def __init__(self):
            self.width, self.height = 0, 0
            self.pixfmt = CameraFrameData.PixelFormat.RGB8
            self.fidx = 0
            self.data_idx = 0

        def getFrameSize(self) -> int:
            return self.width * self.height * \
                CameraFrameData.pixelBytes(self.pixfmt)

    def __init__(self, sensor_id):
        super().__init__(sensor_id)
        self.meta = [] # List[Meta]

    def getHeader(self) -> SensorDataHeader:
        header = SensorDataHeader()
        header.capture_desc = "# camera_frame %d" % (len(self.meta))
        for m in self.meta:
            pixfmt_str = ""
            if m.pixfmt == CameraFrameData.PixelFormat.RGB8:
                pixfmt_str = "rgb8"
            elif m.pixfmt == CameraFrameData.PixelFormat.GRAY16:
                pixfmt_str = "gray16"
            elif m.pixfmt == CameraFrameData.PixelFormat.GRAY8:
                pixfmt_str ="gray8"
            if not pixfmt_str:
                raise ValueError('invalid pixel format')
            header.capture_desc += ' %d %d %s' % (m.width, m.height, pixfmt_str)
        header.field_names ['Timestamp (ns)']
        return header

    def getTextData(self) -> List[str]:
        return [str(self.timestamp_ns)]

    def setMetaFromHeader(self, capture_desc: str) -> bool:
        try:
            tokens = capture_desc.split(' ')
            if tokens[0] != "#":
                raise ValueError('invalid format: ' + capture_desc)
            if tokens[1] != "camera_frame":
                raise ValueError('invalid format: ' + capture_desc)
            num_cameras = int(tokens[2])
            if num_cameras <= 0:
                raise ValueError('invalid num cameras: %d' % (num_cameras))
            self.meta = [CameraFrameData.Meta() for _ in range(num_cameras)]
            j = 3
            data_idx = 0
            for i, m in enumerate(self.meta):
                m.width = int(tokens[j])
                m.height = int(tokens[j + 1])
                pixfmt = tokens[j + 2]
                if pixfmt == "rgb8":
                    m.pixfmt = CameraFrameData.PixelFormat.RGB8
                elif pixfmt == "gray16":
                    m.pixfmt = CameraFrameData.PixelFormat.GRAY16
                elif pixfmt == "gray8":
                    m.pixfmt = CameraFrameData.PixelFormat.GRAY8
                else:
                    m.pixfmt = CameraFrameData.PixelFormat.UNKNOWN
                if m.width <= 0 or m.height <= 0 or \
                   m.pixfmt == CameraFrameData.PixelFormat.UNKNOWN:
                    raise ValueError(
                        'invalid meta %s x %s : %s for cam %d' % (
                            tokens[j], tokens[j + 1], tokens[j + 2], i))
                m.data_idx = data_idx
                data_idx += m.getFrameSize()
                j += 3
        except ValueError as e:
            LOG_ERROR(e.args[0])
            return False
        return True

    def setMetaFromData(self, data: SensorData) -> None:
        self.meta = CameraFrameData(data).meta.copy()

    def setFromTextData(self, text_data: List[str]) -> bool:
        if len(text_data) != 1: return False
        self.timestamp_ns = int(text_data[0])
        return True

    def getImage(self, idx: int) -> np.ndarray | None:
        if idx < 0 or idx > len(self.meta): return None
        meta = self.meta[idx]
        out = self.data[meta.data_idx : meta.data_idx + meta.getFrameSize()]
        if meta.pixfmt == CameraFrameData.PixelFormat.RGB8:
            out = out.reshape((meta.height, meta.width, 3))
        elif meta.pixfmt == CameraFrameData.PixelFormat.GRAY16:
            out = out.view(np.uint16)
            out = out.reshape((meta.height, meta.width))
        else:
            out = out.reshape((meta.height, meta.width))
        return out

    def getAllImages(self) -> List[np.ndarray] | None:
        if not self.meta: return None
        return [self.getImage(i) for i in range(len(self.meta))]

class LidarData(SensorData):
    DATA_TYPE = SensorDataType.LIDAR
    class Format(Enum): SEQ = 0; LOOP = 1

    def __init__(self, sensor_id):
        super().__init__(sensor_id)
        self.num_points = 0
        self.num_blocks = 0
        self.num_channels = 0

    def getHeader(self) -> SensorDataHeader:
        header = SensorDataHeader()
        header.capture_desc = ""
        header.field_names ['Timestamp (ns)', 'Num pts', 'Num blocks']
        return header

    def getTextData(self) -> List[str]:
        return [str(self.timestamp_ns),
            str(self.num_points), str(self.num_blocks)]

    def setMetaFromHeader(self, capture_desc: str) -> bool:
        return True

    def setMetaFromData(self, data: SensorData) -> None:
        data = LidarData(data)
        self.num_points = data.num_points
        self.num_blocks = data.num_channels
        self.num_channels = data.num_channels
        return True

    def setFromTextData(self, text_data: List[str]) -> bool:
        if len(text_data) != 3: return False
        self.timestamp_ns = int(text_data[0])
        self.num_points = int(text_data[1])
        self.num_blocks = int(text_data[2])
        return True

    def points_idx(self): return 0
    def intensities_idx(self): return 3 * self.num_points * 4
    def azimuth_idxs_idx(self): return self.intensities_idx() + self.num_points
    def azimuth_degs_idx(self): return self.azimuth_idxs_idx() + self.num_points

    def points(self) -> np.ndarray:
        point_bytes = 3 * self.num_points * 4
        data = self.data[self.points_idx():]
        out = data[:point_bytes]
        out = out.view(dtype=np.float32)
        out = out.reshape((self.num_points, 3))
        return out

    def intensities(self) -> np.ndarray:
        data = self.data[self.intensities_idx():]
        out = data[:self.num_points]
        return out

    def azimuth_idxs(self) -> np.ndarray:
        data = self.data[self.azimuth_idxs_idx():]
        out = data[:self.num_points]
        return out

    def azimuth_degs(self) -> np.ndarray:
        data = self.data[self.azimuth_degs_idx():]
        out = data[:4 * self.num_blocks]
        out = out.view(dtype=np.float32)
        out = out.reshape((self.num_blocks))
        return out

    # (CVLidar is used in calib-matlab, will be deprecated)
    @staticmethod
    def writeLidarListAsCVLidar(
            data: List[LidarData] | LidarData,
            path: str,) -> bool:
        if type(data) != list:
            data = [data]
        num_seqs = len(data)
        with open(path, 'wb') as f:
            f.write(pack('i', num_seqs))
            for d in data:
                f.write(pack('i', d.num_points))
                d.points().tofile(f)
                d.intensities().tofile(f)
                d.azimuth_idxs().tofile(f)
                f.write(pack('i', d.num_blocks))
                d.azimuth_degs().tofile(f)
                f.write(pack('B', d.num_channels))
                f.write(pack('q', d.timestamp_ns))
        return True

class SensorDataReader:
    def __init__(self, sensor_id: int,
                 data_type: None | SensorDataType = None):
        if data_type:
            self.sensor_id = sensor_id | data_type.value
        else:
            self.sensor_id = sensor_id
        self.header = SensorDataHeader()
        self.read_index_list = []
        self.setReadIndex(0, 1, -1)

    def open(self, filepath_prefix: str,
             capture_bin_fmt: str = '%03d.bin') -> bool:
        try:
            self.filepath_prefix = filepath_prefix
            self.capture_bin_fmt = capture_bin_fmt
            # text file
            self.txt_filepath = filepath_prefix + ".csv"
            self.txt_file = open(self.txt_filepath, 'r')
            if self.txt_file.closed:
                LOG_ERROR(
                    'SensorDataReader::Open: failed to open a text file %s' % (
                        self.txt_filepath))
                return False
            lines = self.txt_file.readlines()
            self.end_idx = len(lines) - 3
            self.txt_file.close()
            self.txt_file = open(self.txt_filepath, 'r')
            self.header.capture_desc = self.txt_file.readline().strip()
            self.header.field_names = \
                self.txt_file.readline().strip().split(',')
            for i in range(0, self.start_read_idx):
                self.txt_file.readline()
            self.cur_read_idx = self.start_read_idx

            # binary file
            bin_filepath = filepath_prefix + capture_bin_fmt % 0
            self.bin_file = open(bin_filepath, 'rb')
            self.packet_size = unpack('i', self.bin_file.read(4))[0]
            self.bin_file.seek(0, 2)
            self.num_packets_per_file = self.bin_file.tell() // \
                (self.packet_size + 4)
            self.bin_file.close()
            self.bin_file_idx = self.start_read_idx // self.num_packets_per_file
            bin_filepath = filepath_prefix + capture_bin_fmt % self.bin_file_idx
            self.bin_file = open(bin_filepath, 'rb')
            rest_read_idx = self.start_read_idx - \
                (self.bin_file_idx * self.num_packets_per_file)
            if rest_read_idx > 0:
                self.bin_file.seek(
                    rest_read_idx * (self.packet_size + 4), 1)
            self.cur_read_idx_in_file = rest_read_idx
            self.data_count = 0
            if self.read_index_list: self.index_list_idx = 0
            return True
        except FileNotFoundError:
            LOG_ERROR('SensorDataReader::open: failed to open capture file')
            return False

    def setReadIndex(self, start: int, step: int, end: int ) -> None:
        if self.read_index_list:
            LOG_WARNING('SensorDataReader::setReadIndex: clear read_index_list')
            self.read_index_list = []
        self.start_read_idx = start
        self.step_read_idx = step
        self.end_read_idx = end

    def setReadIndexList(self, idxs: List[int]) -> None:
        self.read_index_list = sorted(idxs)
        self.index_list_idx = 0
        if not self.read_index_list: return
        self.start_read_idx = self.read_index_list[0]
        self.end_read_idx = self.read_index_list[-1]

        # re-open files
        if not self.txt_file.closed: self.txt_file.close()
        if not self.bin_file.closed: self.bin_file.close()
        self.txt_file = open(self.txt_filepath, 'r')
        for i in range(0, self.start_read_idx + 1):
            self.txt_file.readline()

        self.cur_read_idx = self.start_read_idx
        self.bin_file_idx = self.start_read_idx // self.num_packets_per_file
        bin_filepath = self.filepath_prefix + self.capture_bin_fmt % (
            self.bin_file_idx)
        self.bin_file = open(bin_filepath, 'rb')
        rest_read_idx = self.start_read_idx - \
            (self.bin_file_idx * self.num_packets_per_file)
        if rest_read_idx > 0:
            self.bin_file.seek(
                rest_read_idx * (self.packet_size + 4), 1)
        self.cur_read_idx_in_file = rest_read_idx

    def _readSequential(self) -> SensorData | None:
        if (self.end_read_idx >= 0 and self.cur_read_idx > self.end_read_idx):
            return None
        sensor_type = SensorData.getType(self.sensor_id)
        data = SensorData(self.sensor_id)
        if sensor_type == SensorDataType.CAMERA_FRAME.value:
            data = CameraFrameData(self.sensor_id)
        elif sensor_type == SensorDataType.LIDAR.value:
            data = LidarData(self.sensor_id)
        else:
            LOG_ERROR('SensorDataReader.read: unknown type for %d' %
                (self.sensor_id))
            return None

        if self.read_index_list and \
           self.index_list_idx < len(self.read_index_list) - 1:
            self.step_read_idx = \
                self.read_index_list[self.index_list_idx + 1] - \
                self.read_index_list[self.index_list_idx]
            if self.step_read_idx <= 0:
                LOG_ERROR(
                    'SensorDataReader.read: read_index_list is not sorted')
            else:
                self.index_list_idx += 1

        if not self.txt_file.closed:
            line = self.txt_file.readline()
            text_data = line.strip().split(',')
            if not line or not text_data:
                return None
            data.setFromTextData(text_data)
            for i in range(0, self.step_read_idx - 1):
                self.txt_file.readline()

        if not self.bin_file.closed:
            b = self.bin_file.read(4)
            if len(b) != 4:
                LOG_ERROR('SensorDataReader.read: failed to read bin file')
                self.close()
                return None
            packet_size = unpack('i', b)[0]
            data.data = np.fromfile(self.bin_file,
                dtype=np.uint8, count=packet_size)
            data.setMetaFromHeader(self.header.capture_desc)
            if sensor_type == SensorDataType.CAMERA_FRAME.value:
                for m in data.meta:
                    m.fidx = self.cur_read_idx
            self.data_count += 1

            if self.step_read_idx > 0:
                self.cur_read_idx += self.step_read_idx
                self.cur_read_idx_in_file += self.step_read_idx
                open_next_file = \
                    self.cur_read_idx_in_file >= self.num_packets_per_file
                idx_to_move = self.step_read_idx - 1
                if open_next_file:
                    while self.cur_read_idx_in_file >= \
                            self.num_packets_per_file:
                        self.cur_read_idx_in_file -= self.num_packets_per_file
                        self.bin_file_idx += 1
                    self.bin_file.close()
                    try:
                        bin_filepath = self.filepath_prefix + \
                            self.capture_bin_fmt % self.bin_file_idx
                        self.bin_file = open(bin_filepath, 'rb')
                        idx_to_move = self.cur_read_idx_in_file
                    except:
                        pass
                if idx_to_move > 0 and self.bin_file.seekable():
                    self.bin_file.seek(
                        (self.packet_size + 4) * idx_to_move, 1)
        return data

    def read(self, idx=None) -> SensorData | None:
        if idx is None:
            return self._readSequential()
        if (self.end_read_idx >= 0 and idx > self.end_read_idx):
            return None
        if not self.txt_file.closed:
            LOG_WARNING('txt file may be used in other process')
            self.txt_file.close()
        if not self.bin_file.closed:
            LOG_WARNING('bin file may be used in other process')
            self.bin_file.close()
        sensor_type = SensorData.getType(self.sensor_id)
        data = SensorData(self.sensor_id)
        if sensor_type == SensorDataType.CAMERA_FRAME.value:
            data = CameraFrameData(self.sensor_id)
        elif sensor_type == SensorDataType.LIDAR.value:
            data = LidarData(self.sensor_id)
        else:
            LOG_ERROR('SensorDataReader.read: unknown type for %d' %
                (self.sensor_id))
            return None
        self.txt_file = open(self.txt_filepath, 'r')
        for _ in range(-2, idx): self.txt_file.readline()
        line = self.txt_file.readline()
        text_data = line.strip().split(',')
        self.txt_file.close()
        if not line or not text_data:
            return None
        data.setFromTextData(text_data)
        # binary file
        bin_file_idx = idx // self.num_packets_per_file
        bin_filepath = self.filepath_prefix + self.capture_bin_fmt % (
            bin_file_idx)
        self.bin_file = open(bin_filepath, 'rb')
        rest_read_idx = idx - (bin_file_idx * self.num_packets_per_file)
        if rest_read_idx > 0:
            self.bin_file.seek(
                rest_read_idx * (self.packet_size + 4), 1)
        b = self.bin_file.read(4)
        if len(b) != 4:
            LOG_ERROR('SensorDataReader.read: failed to read bin file')
            self.bin_file.close()
            return None
        packet_size = unpack('i', b)[0]
        data.data = np.fromfile(self.bin_file,
            dtype=np.uint8, count=packet_size)
        self.bin_file.close()
        data.setMetaFromHeader(self.header.capture_desc)
        if sensor_type == SensorDataType.CAMERA_FRAME.value:
            for m in data.meta:
                m.fidx = idx
        return data

    def close(self):
        self.txt_file.close()
        self.bin_file.close()