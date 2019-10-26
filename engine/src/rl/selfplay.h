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
 * @file: selfplay.h
 * Created on 16.09.2019
 * @author: queensgambit
 *
 * Functionality for running CrazyAra in self play mode
 */

#ifndef SELFPLAY_H
#define SELFPLAY_H

#include "../agents/mctsagent.h"
#include "gamepgn.h"
#include "../manager/statesmanager.h"
#include "tournamentresult.h"

#ifdef USE_RL
class SelfPlay
{
private:
    MCTSAgent* mctsAgent;
    GamePGN gamePGN;
    EvalInfo evalInfo;

    /**
     * @brief generate_game Generates a new game in self play mode
     * @param variant Current chess variant
     * @param searchLimits Search limits struct
     */
    void generate_game(Variant variant, SearchLimits& searchLimits);

    /**
     * @brief generate_arena_game Generates a game of the current NN weights vs the new acquired weights
     * @param whitePlayer MCTSAgent which will play with the white pieces
     * @param blackPlayer MCTSAgent which will play with the black pieces
     * @param variant Current chess variant
     * @param searchLimits Search limits struct
     */
    Result generate_arena_game(MCTSAgent *whitePlayer, MCTSAgent *blackPlayer, Variant variant, SearchLimits& searchLimits);

    /**
     * @brief write_game_to_pgn Writes the game log to a pgn file
     */
    void write_game_to_pgn(const std::string& pngFileName);

    /**
     * @brief set_game_result Sets the game result to the gamePGN object
     * @param terminalNode Terminal node of the game
     */
    void set_game_result_to_pgn(const Node* terminalNode);

    /**
     * @brief init_board
     * @param variant
     * @return
     */
    inline Board* init_board(Variant variant);
public:
    SelfPlay(MCTSAgent* mctsAgent);

    /**
     * @brief go Starts the self play game generation for a given number of games
     * @param numberOfGames Number of games to generate
     * @param searchLimits Search limit struct
     */
    void go(size_t numberOfGames, SearchLimits& searchLimits);

    /**
     * @brief go_arena Starts comparision matches between the original mctsAgent with the old NN weights and
     * the mctsContender which uses the new updated wieghts
     * @param mctsContender MCTSAgent using different NN weights
     * @param numberOfGames Number of games to compare
     * @param searchLimits Search limit struct
     * @return Score in respect to the contender, as floating point number.
     *  Wins give 1.0 points, 0.5 for draw, 0.0 for loss.
     */
    TournamentResult go_arena(MCTSAgent *mctsContender, size_t numberOfGames, SearchLimits& searchLimits);
};
#endif

#endif // SELFPLAY_H