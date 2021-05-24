//#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

#include "agent/doomagent.h"


void sleep(unsigned int time){
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
}

using namespace vizdoom;

AgentDebug doomDebug(2);

int main() {  
	
    DoomGame *game = new DoomGame();

    // Sets path to vizdoom engine executive which will be spawned as a separate process.
    game->setViZDoomPath("../../../build/bin/vizdoom"); // Default is "./vizdoom"

    // Sets path to doom2 iwad resource file which contains the actual doom game.
    //game->setDoomGamePath("../../../wad/freedoom2.wad");//Default "./doom2.wad"
    game->setDoomGamePath("../../../wad/doom2.wad");      

    // Sets path to additional resources iwad file which is your scenario iwad.
    // If not specified default doom2 maps will be used and it's pretty much useless... unless you want to play doom.
    //game->setDoomScenarioPath("../../../scenarios/basic.wad");
	game->setDoomMap("map01");    

	game->setScreenResolution(RES_160X120); // Sets resolution. Default is 320X240
    game->setScreenFormat(RGB24); // Default is CRCGCB.
    //game->setRenderHud(false);
    //game->setRenderMinimalHud(false); // If hud is enabled
    //game->setRenderCrosshair(false);
    game->setRenderWeapon(true);
    game->setRenderDecals(false);
    game->setRenderParticles(false);
    //game->setRenderEffectsSprites(false);
    //game->setRenderMessages(false);
    //game->setRenderCorpses(false);

	game->addAvailableButton(MOVE_LEFT);
    game->addAvailableButton(MOVE_RIGHT);
    game->addAvailableButton(ATTACK);
	game->addAvailableButton(MOVE_FORWARD);
	game->addAvailableButton(MOVE_BACKWARD);
	game->addAvailableButton(TURN_LEFT);
	game->addAvailableButton(TURN_RIGHT);
	game->addAvailableButton(USE);

    // Adds game variables that will be included in state.
    game->addAvailableGameVariable(AMMO2);

    // Causes episodes to finish after 200 tics (actions)
    game->setEpisodeTimeout(20000);
    game->setEpisodeStartTime(10);
    game->setWindowVisible(true);
    game->setSoundEnabled(true);
    //game->setConsoleEnabled(true);
    // (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game->setMode(PLAYER);

    // Further configuration won't take any effect from now on.
    game->init();

    // Define some actions. Each list entry corresponds to declared buttons:
    // game->getAvailableButtonsSize() // to check the number of available buttons.
    // more combinations are possible but only 8 are included for transparency.
    //std::vector<double> actions[8];
    //actions[0] = {1, 0, 0, 0, 0, 0, 0, 0};
    //actions[1] = {0, 1, 0, 0, 0, 0, 0, 0};
    //actions[2] = {0, 0, 1, 0, 0, 0, 0, 0};
    //actions[3] = {0, 0, 0, 1, 0, 0, 0, 0};
    //actions[4] = {0, 0, 0, 0, 1, 0, 0, 0};
    //actions[5] = {0, 0, 0, 0, 0, 1, 0, 0};
    //actions[6] = {0, 0, 0, 0, 0, 0, 1, 0};
    //actions[7] = {0, 0, 0, 0, 0, 0, 0, 1};

    std::srand(time(0));
			
    int episodes = 10;
	// Lunar lander agentConf = {x, 50000, 4, 8, x, x, x}.
	AgentConfig agentConf = {8, 1000, 1, 1, 7, 1, 1};
	
	// My old configuration.
	NetworkConfig netConf = {{160, 120, 3}, {8, 4}, {4, 2}, {1, 1}, {RELU, RELU}, {4, 2}, {1, 1}, {50, 20}, 8};				// 2GB RAM, 8sec per step.
	
	// Small configuration from ViZDoom paper experiment A and B.
	//NetworkConfig netConf = {{60, 45, 3}, {32, 32}, {7, 4}, {1, 1}, {RELU, RELU}, {2, 2}, {1, 1}, {800}, 3}; 				// 8GB RAM, 45sec per step.
	//NetworkConfig netConf = {{120, 45, 3}, {32, 32, 32}, {7, 5, 3}, {1, 1, 1}, {RELU, RELU}, {2, 2}, {1, 1}, {1024}, 4};	
	
	// My new configuration.
	//NetworkConfig netConf = {{60, 45, 3}, {32, 32}, {7, 4}, {1, 1}, {RELU, RELU}, {2, 2}, {1, 1}, {20}, 8};				// 3GB RAM, 15sec per step.
	
	OptimizerConfig optConf = {0.1, 0.2, 0.3, 0.5};
	DoomAgent agent(agentConf, netConf, optConf);

    // Pause the engine after each action. Easier to keep track of what's happening
    unsigned int sleepTime = 1000 / DEFAULT_TICRATE; // = 28
    for (int i = 0; i < episodes; ++i) {
        std::cout << "Episode #" << i + 1 << "\n";
		unsigned b = 0;
        //It is not needed right after init(). It doesn't cost much and is nicer.
        game->newEpisode();
		agent.agentStart(game->getState()->screenBuffer);
        while (!game->isEpisodeFinished()) {
			//auto start = std::chrono::high_resolution_clock::now(); 
            GameStatePtr state = game->getState(); //is std::shared_ptr<GameState>
            unsigned int n              = state->number;
            std::vector<double> vars    = state->gameVariables; 
            //BufferPtr is std::shared_ptr<Buffer>, Buffer is std::vector<uint8_t>
			BufferPtr screenBuf         = state->screenBuffer;
            //BufferPtr depthBuf          = state->depthBuffer;
            //BufferPtr labelsBuf         = state->labelsBuffer;
            //BufferPtr automapBuf        = state->automapBuffer;
			//std::vector<Label> labels   = state->labels;

//----------------------------------------------------------------------------//
//			AGENT STEP FUNCTION APPROXIMATOR GOES HERE						//
			std::vector<double> currentAction = agent.agentStep(0.0, screenBuf);
			//for(unsigned j=0; j<current_action.size(); ++j)
				//	std::cout << current_action[j] << " ";
			//std::cout << std::endl;
			//std::vector<double> currentAction = std::vector<double>(8);
			double reward = game->makeAction(currentAction);
//----------------------------------------------------------------------------//
						
			// Make random action and get reward
			//unsigned num_buttons = game->getAvailableButtonsSize();
            //double reward = game->makeAction(actions[std::rand() % num_buttons]); 
            // Makes a "prolonged" action and skip frames.            
			// double reward = game->makeAction(choice(actions), skiprate)

            // The same could be achieved with:
            //game->setAction(choice(actions)) // if PLAYER mode is active
			//game->advanceAction(); // advanceAction(4) Advance few tics at once
			//std::vector<double> lastAction = game->getLastAction();
			//double reward = game->getLastReward()

            //std::cout << "State #" << n << "\n";
            //std::cout << "Game variables: " << vars[0] << "\n";
            //std::cout << "Action reward: " << reward << "\n";
            //std::cout << "=====================\n";

			//auto stop = std::chrono::high_resolution_clock::now();
			//auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			//std::cout << duration.count() << std::endl; 
						
            if(sleepTime) sleep(sleepTime);
        }

        std::cout << "Episode finished.\n";
        std::cout << "Total reward: " << game->getTotalReward() << "\n";
        std::cout << "************************\n";
    }

    // It will be done automatically in destructor but after close You can init it again with different settings.
    game->close();
    delete game;
}
        
