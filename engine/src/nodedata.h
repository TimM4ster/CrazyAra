/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

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
 * @file: nodedata.h
 * Created on 25.04.2020
 * @author: queensgambit
 *
 * Node data is a data container which is unavailable for all nodes <= 1 to reduce memory consumption.
 */

#ifndef NODEDATA_H
#define NODEDATA_H

#include <iostream>
#include <mutex>
#include <unordered_map>

#include <blaze/Math.h>
#include "position.h"
#include "movegen.h"
#include "board.h"

#include "agents/config/searchsettings.h"
#include "constants.h"

using blaze::HybridVector;
using blaze::DynamicVector;
using namespace std;


enum NodeType : uint8_t {
    SOLVED_WIN,
    SOLVED_DRAW,
    SOLVED_LOSS,
    UNSOLVED
};

class Node;

/**
 * @brief The NodeData struct stores the member variables for all expanded child nodes which have at least been visited two times
 */
struct NodeData
{
    DynamicVector<float> childNumberVisits;
    DynamicVector<float> actionValues;
    DynamicVector<float> qValues;
    vector<Node*> childNodes;

    float terminalVisits;

    uint16_t checkmateIdx;
    uint16_t endInPly;
    uint16_t noVisitIdx;
    uint16_t numberUnsolvedChildNodes;

    NodeType nodeType;
    NodeData(size_t numberChildNodes);

    auto get_q_values();

public:
    /**
     * @brief add_empty_node Adds a new empty node to its child nodes
     */
    void add_empty_node();

    /**
     * @brief reserve_initial_space Reserves memory for PRESERVED_ITEMS number of child nodes
     */
    void reserve_initial_space();
};


#endif // NODEDATA_H