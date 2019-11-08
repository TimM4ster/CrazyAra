/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018  Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: traindataexporter.cpp
 * Created on 12.09.2019
 * @author: queensgambit
 */

#ifdef USE_RL
#include "traindataexporter.h"
#include <inttypes.h>

void TrainDataExporter::export_pos(const Board *pos, const EvalInfo& eval, size_t idxOffset)
{
    if (startIdx+idxOffset >= numberSamples) {
        cout << "string info Extended number of maximum samples" << endl;
        return;
    }
    export_planes(pos, idxOffset);
    export_policy(eval.legalMoves, eval.policyProbSmall, pos->side_to_move(), idxOffset);
    // value will be set later in export_game_result()
}

void TrainDataExporter::export_best_move_q(const EvalInfo &eval, size_t idxOffset)
{
    if (startIdx+idxOffset >= numberSamples) {
        cout << "string info Extended number of maximum samples" << endl;
        return;
    }
    // Q value of "best" move (a.k.a selected move after mcts search)
    // write value to roi
    z5::types::ShapeType offsetValue = { startIdx+idxOffset };
    xt::xarray<float> qArray({ 1 }, eval.bestMoveQ);

    z5::multiarray::writeSubarray<float>(dbestMoveQ, qArray, offsetValue.begin());
}

void TrainDataExporter::export_game_result(const int16_t result, size_t idxOffset, size_t plys)
{
    if (startIdx+idxOffset >= numberSamples) {
        cout << "string info Extended number of maximum samples" << endl;
        return;
    }
    if (startIdx+idxOffset+plys > numberSamples) {
        plys -= startIdx+idxOffset+plys - numberSamples;
        cout << "string info Adjust samples to export to " << plys << endl;
    }

    // value
    // write value to roi
    z5::types::ShapeType offsetValue = { startIdx+idxOffset };
    xt::xarray<int16_t>::shape_type shapeValue = { plys };
    xt::xarray<int16_t> valueArray(shapeValue, result);

    if (result != DRAW) {
        // invert the result on every second ply
        for (size_t idx = 1; idx < plys; idx+=2) {
            valueArray.data()[idx] = -result;
        }
    }

    z5::multiarray::writeSubarray<int16_t>(dValue, valueArray, offsetValue.begin());
    startIdx += plys;
    gameIdx++;
    export_start_idx();
}

TrainDataExporter::TrainDataExporter(const string& fileName, const string& deviceName, size_t numberChunks, size_t chunkSize):
    numberChunks(numberChunks),
    chunkSize(chunkSize),
    numberSamples(numberChunks * chunkSize),
    gameIdx(0),
    startIdx(0)
{
    // get handle to a File on the filesystem
    z5::filesystem::handle::File file(fileName);

    fileNameGameIdx = string("gameIdx_") + deviceName + string(".txt");
    fileNameStartIdx = string("startIdx_") + deviceName + string(".txt");

    if (file.exists()) {
        open_dataset_from_file(file);
    }
    else {
        create_new_dataset_file(file);
    }
}

size_t TrainDataExporter::get_number_samples() const
{
    return numberSamples;
}

bool TrainDataExporter::is_file_full()
{
    return startIdx >= numberSamples;
}

void TrainDataExporter::export_planes(const Board *pos, size_t idxOffset)
{
    // x / plane representation
    float inputPlanes[NB_VALUES_TOTAL];
    board_to_planes(pos, pos->number_repetitions(), false, inputPlanes);
    // write array to roi
    z5::types::ShapeType offsetPlanes = { startIdx+idxOffset, 0, 0, 0 };
    xt::xarray<int16_t>::shape_type policyShape = { 1, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH };
    xt::xarray<int16_t> policy(policyShape);
    for (size_t idx = 0; idx < NB_VALUES_TOTAL; ++idx) {
        policy.data()[idx] = int16_t(inputPlanes[idx]);
    }
    z5::multiarray::writeSubarray<int16_t>(dx, policy, offsetPlanes.begin());
}

void TrainDataExporter::export_policy(const vector<Move>& legalMoves, const DynamicVector<float>& policyProbSmall, Color sideToMove, size_t idxOffset)
{
    assert(legalMoves.size() == policyProbSmall.size());

    // write array to roi
    z5::types::ShapeType offsetPolicy = { startIdx+idxOffset, 0 };
    xt::xarray<float>::shape_type shapePolicy = { 1, NB_LABELS };
    xt::xarray<float> policy(shapePolicy, 0);

    for (size_t idx = 0; idx < legalMoves.size(); ++idx) {
        size_t policyIdx;
        if (sideToMove == WHITE) {
            policyIdx = MV_LOOKUP_CLASSIC[legalMoves[idx]];
        }
        else {
            policyIdx = MV_LOOKUP_MIRRORED_CLASSIC[legalMoves[idx]];
        }
        policy[policyIdx] = policyProbSmall[idx];
    }
    z5::multiarray::writeSubarray<float>(dPolicy, policy, offsetPolicy.begin());

}

void TrainDataExporter::export_start_idx()
{
    // gameStartIdx
    // write value to roi
    z5::types::ShapeType offsetStartIdx = { gameIdx };
    xt::xarray<int32_t> arrayGameStartIdx({ 1 }, int32_t(startIdx));
    z5::multiarray::writeSubarray<int32_t>(dStartIndex, arrayGameStartIdx, offsetStartIdx.begin());

    ofstream startIdxFile;
    startIdxFile.open(fileNameStartIdx);
    // set the next startIdx to continue
    startIdxFile << startIdx;
    startIdxFile.close();
    ofstream gameIdxFile;
    gameIdxFile.open(fileNameGameIdx);
    gameIdxFile << gameIdx;
    gameIdxFile.close();
}

void TrainDataExporter::open_dataset_from_file(const z5::filesystem::handle::File& file)
{
    dStartIndex = z5::openDataset(file,"start_indices");
    dx = z5::openDataset(file,"x");
    dValue = z5::openDataset(file,"y_value");
    dPolicy = z5::openDataset(file,"y_policy");
    dbestMoveQ = z5::openDataset(file, "y_best_move_q");
    ifstream startIdxFile;
    startIdxFile.open(fileNameStartIdx);
    startIdxFile >> startIdx;
    startIdxFile.close();
    ifstream gameIdxFile;
    gameIdxFile.open(fileNameGameIdx);
    gameIdxFile >> gameIdx;
    gameIdxFile.close();
}

void TrainDataExporter::create_new_dataset_file(const z5::filesystem::handle::File &file)
{
    // create the file in zarr format
    const bool createAsZarr = true;
    z5::createFile(file, createAsZarr);

    // create a new zarr dataset
    std::vector<size_t> shape = { numberSamples, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH };
    std::vector<size_t> chunks = { chunkSize, NB_CHANNELS_TOTAL, BOARD_HEIGHT, BOARD_WIDTH };
    dStartIndex = z5::createDataset(file, "start_indices", "int32", { numberSamples }, { chunkSize });
    dx = z5::createDataset(file, "x", "int16", shape, chunks);
    dValue = z5::createDataset(file, "y_value", "int16", { numberSamples }, { chunkSize });
    dPolicy = z5::createDataset(file, "y_policy", "float32", { numberSamples, NB_LABELS }, { chunkSize, NB_LABELS });
    dbestMoveQ = z5::createDataset(file, "y_best_move_q", "float32", { numberSamples }, { chunkSize });

    export_start_idx();
}
#endif
