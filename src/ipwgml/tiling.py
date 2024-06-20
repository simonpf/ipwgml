"""
ipwgml.tiling
=============

Provides functionality for tiling input data and assembling tiled results.
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr


def get_starts_and_clips(
        extent: int,
        tile_size: int,
        overlap: int
) -> Tuple[List[int], List[int]]:
    """
    Calculate start indices and numbers of clipped pixels for a given
    side length, tile size and overlap.

    Args:
        extent: The extent of the dimension to tile.
        tile_size: The size of each tile.
        overlap: The number of pixels of overlap.
        soft_end: Allow the last tile to go beyond ``n``, see notes for details

    Return:
        A tuple ``(start, clip)`` containing the start indices of each tile
        and the number of pixels to clip between each neighboring tiles.
    """
    starts = []
    clips = []
    start = 0
    while start + tile_size < extent:
        starts.append(start)
        if start > 0:
            clips.append(overlap // 2)
        start = start + tile_size - overlap
    starts.append(max(extent - tile_size, 0))
    if len(starts) > 1:
        clips.append((starts[-2] + tile_size - starts[-1]) // 2)
    return starts, clips


class DatasetTiler:
    """
    This tiler provides functionality for tiling an xarray.Dataset into equal-sized tiles.
    """
    def __init__(
            self,
            dataset: xr.Dataset,
            tile_size: int | None = 512,
            overlap: int = 32,
            spatial_dims: Optional[Tuple[str, str]] = None
    ):
        """
        Args:
            dataset: List of input tensors for the retrieval.
            tile_size: The size of a single tile. If this is None the tiler
                returns a single tile extending over the full spatial extent of the dataset.
            overlap: The overlap between two subsequent tiles.
            spatial_dims: A tuple containing the names of the spatial dimensions along which
                to tile the dataset. If not set, will use the two lattermost dimensions
                in the datset.
        """
        self.dataset = dataset
        if spatial_dims is None:
            spatial_dims = list(dataset.dims)[-2:]
        self.spatial_dims = spatial_dims
        rows, cols = [dataset[dim].size for dim in spatial_dims]
        self.n_rows = rows
        self.n_cols = cols

        if tile_size is None:
            tile_size = (dataset[spatial_dims[0]].size, dataset[spatial_dims[1]].size)

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        if len(tile_size) == 1:
            tile_size = tile_size * 2
        self.tile_size = tile_size
        self.overlap = overlap

        min_len = min(self.tile_size[0], self.tile_size[1])
        if overlap > min_len // 2:
            raise ValueError("Overlap must not exceed the half of the tile size.")

        row_starts, row_clips = get_starts_and_clips(self.n_rows, tile_size[0], overlap)
        self.row_starts = row_starts
        self.row_clips = row_clips
        col_starts, col_clips = get_starts_and_clips(self.n_cols, tile_size[1], overlap)
        self.col_starts = col_starts
        self.col_clips = col_clips

        self.n_rows_tiled = len(self.row_starts)
        self.n_cols_tiled = len(self.col_starts)

        if self.n_rows < tile_size[0]:
            left = (tile_size[0] - self.n_rows) // 2
            right = tile_size[0] - self.n_rows - left
            self.row_pad = (left, right)
        else:
            self.row_pad = None

        if self.n_cols < tile_size[1]:
            left = (tile_size[1] - self.n_cols) // 2
            right = tile_size[1] - self.n_cols - left
            self.col_pad = (left, right)
        else:
            self.col_pad = None

    def get_tile(self, row_ind: int, col_ind: int) -> xr.Dataset:
        """
        Get tile in the 'row_ind'th row and 'col_ind'th column of the two
        dimensional tiling.

        Args:
            row_ind: The 0-based row index of the tile.
            col_ind: The 0-based column index of the tile.

        Return:
            An xarray.Dataset containing the requested tile.
        """
        row_start = self.row_starts[row_ind]
        col_start = self.col_starts[col_ind]
        slices = {
            self.spatial_dims[0]: slice(row_start, row_start + self.tile_size[0]),
            self.spatial_dims[1]: slice(col_start, col_start + self.tile_size[1]),
        }
        return self.dataset[slices]


    def get_slices(self, row_ind: int, col_ind) -> Tuple[slice, slice]:
        """
        Return slices for the clipping of the tiles.

        Args:
            row_ind: The 0-based row index of the tile.
            col_ind: The 0-based column index of the tile.

        Return:
            Tuple of slices that can be used to clip the retrieval
            results to obtain non-overlapping tiles.
        """
        if row_ind == 0:
            row_clip_l = 0
        else:
            row_clip_l = self.row_clips[row_ind - 1]
        if row_ind >= self.n_rows_tiled - 1:
            row_clip_r = self.tile_size[0]
        else:
            row_clip_r = self.tile_size[0] - self.row_clips[row_ind]
        slice_row = slice(row_clip_l, row_clip_r)

        if col_ind == 0:
            col_clip_l = 0
        else:
            col_clip_l = self.col_clips[col_ind - 1]
        if col_ind >= self.n_cols_tiled - 1:
            col_clip_r = self.tile_size[1]
        else:
            col_clip_r = self.tile_size[1] - self.col_clips[col_ind]
        slice_col = slice(col_clip_l, col_clip_r)

        return {
            self.spatial_dims[0]: slice_row,
            self.spatial_dims[1]: slice_col
        }


    def get_weights(
            self,
            row_ind: int,
            col_ind,
            like: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get weights to reassemble results.

        Args:
            row_ind: Row-index of the tile.
            col_ind: Column-index of the tile.
            like: An optional numpy.ndarray or torch.Tensor. If it is a torch.Tensor, the weights
                will use the same dtype and be created on the same device as the tensor.

        Return:
            Numpy array or torch tensor containing weights for the corresponding tile.
        """
        n_rows, n_cols = self.tile_size
        w_rows = np.ones((n_rows, n_cols))

        if row_ind > 0:
            trans_start = self.row_starts[row_ind]
            if row_ind > 1:
                trans_end_prev = self.row_starts[row_ind - 2] + self.tile_size[0]
                trans_start = max(trans_start, trans_end_prev)
            zeros = trans_start - self.row_starts[row_ind]
            trans_end = self.row_starts[row_ind - 1] + self.tile_size[0]
            # Limit transition zone to overlap.
            l_trans = min(trans_end - trans_start, self.overlap)
            w_rows[:zeros] = 0.0
            w_rows[zeros : zeros + l_trans] = np.linspace(0, 1, l_trans)[..., np.newaxis]

        if row_ind < self.n_rows_tiled - 1:
            trans_start = self.row_starts[row_ind + 1]
            trans_end = self.row_starts[row_ind] + self.tile_size[0]
            l_trans = min(trans_end - trans_start, self.overlap)

            start = trans_start - self.row_starts[row_ind]
            w_rows[start : start + l_trans] = np.linspace(1, 0, l_trans)[..., np.newaxis]
            w_rows[start + l_trans :] = 0.0

        w_cols = np.ones((n_rows, n_cols))
        if col_ind > 0:
            trans_start = self.col_starts[col_ind]
            if col_ind > 1:
                trans_end_prev = self.col_starts[col_ind - 2] + self.tile_size[1]
                trans_start = max(trans_start, trans_end_prev)
            zeros = trans_start - self.col_starts[col_ind]
            trans_end = self.col_starts[col_ind - 1] + self.tile_size[1]
            l_trans = min(trans_end - trans_start, self.overlap)
            w_cols[:, :zeros] = 0.0
            w_cols[:, zeros : zeros + l_trans] = np.linspace(0, 1, l_trans)[np.newaxis]

        if col_ind < self.n_cols_tiled - 1:
            trans_start = self.col_starts[col_ind + 1]
            if col_ind > 0:
                trans_end_prev = self.col_starts[col_ind - 1] + self.tile_size[1]
                trans_start = max(trans_start, trans_end_prev)
            trans_end = self.col_starts[col_ind] + self.tile_size[1]
            l_trans = min(trans_end - trans_start, self.overlap)

            start = trans_start - self.col_starts[col_ind]
            w_cols[:, start : start + l_trans] = np.linspace(1, 0, l_trans)[np.newaxis]
            w_cols[:, start + l_trans :] = 0.0

        return w_rows * w_cols

    def assemble(self, tiles):
        """
        Assemble slices back to original shape using linear interpolation in
        overlap regions.

        Args:
            tiles: A list of lists containing the results for each tile.

        Return:
            The data in 'tiles' reassembled along the last to dimensions to match the
            initial dimenions of the input.
        """
        tile_0 = tiles[0][0]
        results = self.initialize_results(tile_0)
        for i, row in enumerate(tiles):
            for j, tle in enumerate(row):
                self.assemble_tile(i, j, results, tle)
        return results


    def initialize_results(self, results_t):
        """
        Initialize containers for assembled results from the results from
        the first tile.

        Args:
            results_t: Retrieval results returned from the first tile.

        Return:
            Depending of the structure of 'results_t', a single numpy.ndarray,
            or a (potentially nested) list or dict of numpy.ndarrays.
        """
        if isinstance(results_t, list):
            return [self.initialize_results(res) for res in results_t]
        if isinstance(results_t, tuple):
            return tuple([self.initialize_results(res) for res in results_t])
        if isinstance(results_t, dict):
            return {
                key: self.initialize_results(val)
                for key, val in results_t.items()
            }
        res = results_t

        ds_row = self.tile_size[0] / res.shape[-2]
        ds_col = self.tile_size[1] / res.shape[-1]
        shape = res.shape[:-2] + (int(self.m / ds_row), int(self.n / ds_col))
        if isinstance(res, torch.Tensor):
            return torch.zeros(shape, dtype=res.dtype, device=res.device)
        return np.zeros(shape, dtype=res.dtype)


    def assemble_tile(self, row_index, col_index, results, results_t):
        """
        Assembles results from a single tile into the assembled result
        containers in 'results'.

        Args:
            row_index: The row index identifying the current tile.
            col_index: The column index identifying the current tile.
            results: Container for the assembled results.
            results_t: Results for the current tile.

        """
        if isinstance(results, (list, tuple)):
            assembled = []
            for res, res_t in zip(results, results_t):
                assembled.append(
                    self.assemble_tile(row_index, col_index, res, res_t)
                )
            if isinstance(results, tuple):
                return tuple(assembled)
            return assembled
        if isinstance(results, dict):
            assembled = {}
            for key in results_t.keys():
                res = results[key]
                res_t = results_t[key]
                assembled[key] = self.assemble_tile(
                    row_index,
                    col_index,
                    res,
                    res_t
                )
            return assembled


        ds_row = self.tile_size[0] // results_t.shape[-2]
        ds_col = self.tile_size[1] // results_t.shape[-1]

        i_start = self.i_start[row_index]
        i_end = i_start + self.tile_size[0]
        row_slice = slice(i_start // ds_row, i_end // ds_row)
        j_start = self.j_start[col_index]
        j_end = j_start + self.tile_size[1]
        if self.N == 1:
            j_end = min(self.n, j_end)

        # modulo self.n in case self.wrap_columns is True
        col_slice = (
            np.arange(j_start // ds_col, j_end // ds_col) %
            (self.n // ds_col)
        )

        wgts = self.get_weights(row_index, col_index, like=results_t)[..., ::ds_row, ::ds_col]

        if self.i_pad is not None:
            i_pad = slice(self.i_pad[0] // ds_row, -self.i_pad[-1] // ds_row)
        else:
            i_pad = slice(0, None)

        if self.j_pad is not None:
            if self.wrap_columns:
                j_pad = slice(0, -sum(self.j_pad))
            else:
                j_pad = slice(self.j_pad[0] // ds_col, -self.j_pad[-1] // ds_col)
        else:
            j_pad = slice(0, None)

        results[..., row_slice, col_slice] += wgts[..., i_pad, j_pad] * results_t[..., i_pad, j_pad]


    def __iter__(self):

        results = None

        for row_ind in range(self.M):
            for col_ind in range(self.N):

                results_t = yield self.get_tile(row_ind, col_ind)
                if results_t is None:
                    raise ValueError(
                        " Tile received results that are 'None'. You need to "
                        "provide send results for each tile into the tiler "
                        "iterator using 'send'."
                    )

                if results is None:
                    results = self.initialize_results(results_t)
                self.assemble_tile(row_ind, col_ind, results, results_t)

        return results

    def predict(self, predict_fun):
        """
        Applies a prediction function to all tiles in the input
        and assembles the results.

        Args:
            predict_fun: A callable that takes the input from a single tile
                and returns the corresponding predicted results.

        Return:
            The tile-wise results from 'predict_fun' assembled to the original
            size.
        """
        tiler = iter(self)
        x_t = next(tiler)
        try:
            while True:
                results_t = predict_fun(x_t)
                x_t = tiler.send(results_t)
        except StopIteration as exc:
            results = exc.value
        return results


    def __repr__(self):
        return f"Tiler(tile_size={self.tile_size}, overlap={self.overlap})"


def calculate_padding(tensor, multiple_of=32):
    """
    Calculate torch padding dimensions required to pad the input tensor
    to a multiple of 32.

    Args:
        tensor: The tensor to pad.
        multiple_of: Integer of which the spatial dimensions of 'tensor'
            should be a multiple of.

    Return
        A tuple ``(p_l_n, p_r_n, p_l_m, p_r_m)`` containing the
        left and right padding  for the second to last dimension
        (``p_l_m, p_r_m``) and for the last dimension (``p_l_n, p_r_n``).
    """
    shape = tensor.shape

    n = shape[-1]
    d_n = ceil(n / multiple_of) * multiple_of - n
    p_l_n = d_n // 2
    p_r_n = d_n - p_l_n

    m = shape[-2]
    d_m = ceil(m / multiple_of) * multiple_of - m
    p_l_m = d_m // 2
    p_r_m = d_m - p_l_m
    return (p_l_n, p_r_n, p_l_m, p_r_m)
