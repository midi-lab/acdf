## Annotated Diff
Here we present a full annotated diff of the changes from the python files in ``gridworld_exploration`` directory of the released code of Lamb et. al. 2022. Note that two new files were also added, which implement the periodic evaluation enviromnent: ``periodic_cart_builder.py`` and ``periodic_cart_env.py``.

#### Diff for `buffer.py`
This file handles the transition buffer. Changes are needed to return the 1-step next observation, in addition to the k-step next observation (i.e, x_t, x_{t+1} and x_{t+k}), so that we can use the latent multistep inverse loss. Also fixes a bug in the original Lamb et al. code affecting the use of random training policies.
- Setup for returning x_{t+1} observation
```diff
@@ -79 +79 @@
-    def sample_batch(self, bs, indlst=None, klim=None, only_train_random=True): 
+    def sample_batch(self, bs, indlst=None, klim=None, only_train_random=True, return_immediate=False): 
@@ -85,0 +86,2 @@
+        if (return_immediate):
+            bx_immediate = []
```
- Bug fix for random policies
```diff
@@ -134,0 +139,2 @@
+            if (self.args.policy_selection == 'random'):
+                randk = random.randint(1, maxk)
```
- Changes to return x_{t+1} observation
```diff
@@ -143,0 +150,2 @@
+                if (return_immediate):
+                    _, _, _, x_imm, _, _, _, _, _, _, _  = self.sample_ex(j + 1)
@@ -148 +156 @@
-                if step >= step_n:
+                if step >= step_n: # This should also cover x_immediate, which is between x and x_n
@@ -152,0 +161,2 @@
+                if (return_immediate):
+                    x_immediate = x_imm
@@ -198,0 +209,2 @@
+                if (return_immediate):
+                    bx_immediate.append(x_immediate.cuda())
@@ -211,0 +224,2 @@
+                if (return_immediate):
+                    bx_immediate.append(x_immediate)
@@ -226,0 +241,2 @@
+            if (return_immediate):
+                bx_immediate = torch.cat(bx_immediate, dim=0).cuda()
@@ -240,0 +257,2 @@
+            if (return_immediate):
+                bx_immediate = torch.cat(bx_immediate, dim=0)
@@ -250,2 +268,4 @@
-
-        return ba, by1, by1_, bx, bx_, bv, bk, bd, bg, bpred_y1, bpred_y1_
+        if (return_immediate):
+            return ba, by1, by1_, bx, bx_, bv, bk, bd, bg, bpred_y1, bpred_y1_, bx_immediate
+        else:
+            return ba, by1, by1_, bx, bx_, bv, bk, bd, bg, bpred_y1, bpred_y1_
```

#### Diff for `grid_4room_env.py`
Defines the *baseline* four-room environment.

- Change needed for compatibility with new evaluation method
```diff
@@ -25 +25,2 @@
-        
+        self.total_states = (args.rows)*(args.cols)
+
```
- Fixes a bug in the original Lamb et al. code where the starting state of the agent could be inside of a wall of the maze
```diff
@@ -33,0 +35,22 @@
+    def sample_initial_correct(self): # prevents sampling initial state in walls
+        middle = int(self.rows / 2)
+        quarter = int(self.rows / 4)
+        pos = random.randint(1,  (self.rows-3)**2 + 4 ) # randint is inclusive
+        if pos <= (self.rows-3)**2:
+            pos_x = (pos-1)%(self.rows-3)
+            pos_y =  (pos-1)//(self.rows-3)
+            pos_x += 1
+            if (pos_x >= middle):
+                pos_x += 1
+            pos_y += 1
+            if (pos_y >= middle):
+                pos_y += 1
+            return (pos_x, pos_y)
+        elif pos ==  (self.rows-3)**2 + 1:
+            return (middle, quarter)
+        elif pos ==  (self.rows-3)**2 + 2:
+            return (quarter, middle)
+        elif pos ==  (self.rows-3)**2 + 3:
+            return (middle, self.rows-quarter -1)
+        else:
+            return (self.rows-quarter - 1, middle)
@@ -62,2 +85,2 @@
```
- We start all of the distractor mazes at random states, rather than a single deterministic state
```diff
-            start1 = (random.randint(1,rows-1), random.randint(1,cols-1))
-            start2 = (random.randint(1,rows-1), random.randint(1,cols-1))
+            # start1 = (random.randint(1,rows-1), random.randint(1,cols-1))
+            start1 = self.sample_initial_correct()
@@ -66 +88,0 @@
-            start2 = (1,1)
@@ -76 +98,5 @@
-            self.exo_grids.append(GridWorld(rows, cols, start=(1,1), goal=(rows-1,cols-1)))
+            if self.random_start:
+               starti = self.sample_initial_correct()
+            else:
+                starti = (1,1)
+            self.exo_grids.append(GridWorld(rows, cols, start=starti, goal=(rows-1,cols-1)))
```
#### Diff for `main.py`
Main file, defines the main training loop. Major changes include adding the latent forward model loss, and implementing the new open-loop planning evaluation procedure.

- Add new imports and new command-line arguments, and let the random seed be set by command-line
```diff
@@ -7,2 +6,0 @@
-random.seed(40)
-torch.manual_seed(40)
@@ -25,0 +24,3 @@
+from scipy.sparse import csr_matrix
+from scipy.sparse.csgraph import dijkstra
+
@@ -30 +31 @@
-parser.add_argument('--data', type=str, choices=[ 'mnist', 'maze', 'vis-maze', 'minigrid', 'cartpole'])
+parser.add_argument('--data', type=str, choices=[ 'mnist', 'maze', 'periodic-cart','vis-maze', 'minigrid', 'cartpole'])
@@ -79,0 +81,2 @@
+parser.add_argument('--half_period', type=int, default=5, help='periodicity')
+
@@ -91,0 +95,6 @@
+parser.add_argument("--use_forward", action="store_true", default=False, help='whether to use a forward dynamics loss')
+
+parser.add_argument('--forward_loss_freq', type=int, default=5) 
+
+parser.add_argument("--use_best_model", action="store_true", default=False, help='assess performance of best model from moving average loss')
+
@@ -116,0 +126,6 @@
+parser.add_argument('--seed', type=int, default=0)
+
+parser.add_argument('--log_eval_prefix', type=str)
+
+parser.add_argument('--eval_iter', type=int, default=1000) # Should be multiple of model_train_iter
+
@@ -119,0 +135,2 @@
+parser.add_argument('--no_reset_actions', action='store_true')
+
@@ -126,0 +144,9 @@
+random.seed(args.seed)
+torch.backends.cudnn.deterministic = True
+torch.backends.cudnn.benchmark = False
+torch.manual_seed(args.seed)
+torch.cuda.manual_seed_all(args.seed)
+np.random.seed(args.seed)
+
+if (args.no_reset_actions):
+    args.reset_actions = False
```
- Make environment for evaluation.
```diff
@@ -150,0 +177 @@
+    myenv_eval = Env(args, stochastic_start=stochastic_start)
@@ -152,0 +180,7 @@
```
- Changes needed for new periodic environment
```diff
+elif args.data == 'periodic-cart':
+    #% ------------------ Define MDP as Periodic Track ------------------
+    from periodic_cart_env import PeriodicCartEnv
+    myenv = PeriodicCartEnv(args, stochastic_start=stochastic_start)
+    myenv_eval = PeriodicCartEnv(args, stochastic_start=stochastic_start)
+    env_name = 'periodic-cart'
+
@@ -263 +297 @@
-        if args.data == 'maze':
+        if args.data == 'maze' or args.data == 'periodic-cart':
@@ -282 +316 @@
-        elif args.data=='maze':
+        elif args.data=='maze' or args.data == 'periodic-cart':
```
- Setup for keeping track of the lowest-loss version of the model to evaluate.
```diff
@@ -350 +384,3 @@
-
+losses= []
+update_count = 0
+best_avg_loss = float('inf')
```
- *This is the key point that differentiates ACDF from AC-State*: the point in the training loop where the latent forward loss is actually used:
```diff
@@ -366 +402,4 @@
-    a1, y1, y1_, x_last, x_new, valid_score, k_offset, depth_to_goal, goal_obs_x, embed_pred_y1, embed_pred_y1_ = mybuffer.sample_batch(bs, batch_ind, klim=klim)
+    if (args.use_forward):
+        a1, y1, y1_, x_last, x_new, valid_score, k_offset, depth_to_goal, goal_obs_x, embed_pred_y1, embed_pred_y1_,x_imm = mybuffer.sample_batch(bs, batch_ind, klim=klim, return_immediate=True)
+    else:
+        a1, y1, y1_, x_last, x_new, valid_score, k_offset, depth_to_goal, goal_obs_x, embed_pred_y1, embed_pred_y1_ = mybuffer.sample_batch(bs, batch_ind, klim=klim)
@@ -373 +411,0 @@
-
@@ -375,0 +414,3 @@
+        if (args.use_forward):
+            with torch.no_grad():
+                _,_,ind_imm = model.enc(x_imm.detach(), do_quantize, reinit_codebook,k=k_ind)
@@ -416,0 +458,13 @@
+
+
+        if (args.use_forward):
+            act = torch.nn.functional.one_hot(a1.detach(), num_classes=10)
+            context = torch.cat((z1, act), -1)
+            global update_count
+            if update_count % args.forward_loss_freq == 0:
+                out_2 = model.out_fwd(context)
+            else:
+                out_2 = model.out_fwd(context.detach())
+            loss_fwd = ce(out_2,ind_imm.detach().flatten())
+            loss += (loss_fwd).mean()
+            update_count += 1
```
- Change needed for new periodic environment:
```diff
@@ -528 +582 @@
-        elif args.data == 'maze':
+        elif args.data == 'maze' or 'periodic-cart':
```
- Bug fix for random policies
```diff
@@ -588,4 +642,2 @@
-
-        print('\t', 'step', env_iteration)
-        # curr_episode = mybuffer[-1]
-        # curr_episode.add_example(x, a1, y1, learnable_action = True)
+        if (always_random):
+            steps_to_goal = args.k_steps  # Needs to be set here to be reachable
```
- Change needed for new periodic environment:
```diff
@@ -685 +737 @@
-    elif args.data == 'maze':
+    elif args.data == 'maze' or 'periodic-cart':
```
- Remove excessive print-logging, and make state visitation logging optional
```diff
@@ -752 +804 @@
-    transition.print_codes(init_state.item(), next_state.item(), a1, g_dp)
+    # transition.print_codes(init_state.item(), next_state.item(), a1, g_dp)
@@ -770 +822,2 @@
-        logger.save_state_visits(all_state_visits, env_iteration)
+        if (logger is not None):
+            logger.save_state_visits(all_state_visits, env_iteration)
```
- Keep track of the lowest-loss version of the model to evaluate.
```diff
@@ -795,0 +849,8 @@
+        if (args.use_best_model and iteration >= num_iter*.1):
+            losses.append(loss.detach())
+            if(len(losses) > 20):
+                losses = losses[-20:]
+                if(sum(losses)/len(losses) < best_avg_loss):
+                    best_avg_loss = sum(losses)/len(losses)
+                    print('New best avg loss: ' +str(best_avg_loss))
+                    torch.save(net, args.log_eval_prefix + "_it_"+ str(mybuffer.num_ex) + "_seed_" + str(args.seed) + "_checkpoint.pth")
```
- New procedure to valuate trained model based on success in open-loop planning.
```diff
@@ -818,0 +880,92 @@
+        ################## New final Evaluator ##################
+        # Build graph from current encoder for entire replay buffer
+        if ((iteration+1) % args.eval_iter == 0):
+            if (args.use_best_model):
+                eval_net = torch.load( args.log_eval_prefix + "_it_"+ str(mybuffer.num_ex) + "_seed_" + str(args.seed) + "_checkpoint.pth")
+            else:
+                eval_net = net
+            codes = []
+            codes_ = []
+            eval_net.eval()
+            with torch.no_grad():
+                for x in mybuffer.x:
+                    codes.append(eval_net.encode((x*1.0).cuda())[0].item())
+                for x_ in mybuffer.x_:
+                    codes_.append(eval_net.encode((x_*1.0).cuda())[0].item())
+                    
+                transition_freq_mat = np.zeros([transition.na, args.ncodes, args.ncodes])
+                codes_by_gt = np.zeros([args.ncodes,myenv.total_states]) 
+                for i in range(len(mybuffer.x)):
+                    transition_freq_mat[mybuffer.a[i],codes[i],codes_[i]] += 1
+                    codes_by_gt[codes[i], mybuffer.y1[i]] += 1
+
+                transitions_for_dijkstra = -1 * np.ones([args.ncodes, args.ncodes],dtype=int) # equals an (arbitrary) action if one is available between two codes, -1 otherwise
+
+                for i in range(args.ncodes):
+                    for j in range(transition.na):
+                        if (transition_freq_mat[j,i].sum() !=0):
+                            k = transition_freq_mat[j,i].argmax()
+                            transitions_for_dijkstra[i,k] = j
+
+
+                dijkstra_adjacency_matrix = np.ones([args.ncodes, args.ncodes])
+                dijkstra_adjacency_matrix[transitions_for_dijkstra == -1] = np.inf
+                dijkstra_adjacency_matrix = csr_matrix(dijkstra_adjacency_matrix)
+
+                # For q samples, get two random observations, plan with dijkstras. If it excecutes a correct path, we score
+                q = 1000
+                #NOTE: must set stochastic start
+                wins = 0
+
+                assert args.exo_noise == 'two_maze'
+                assert args.stochastic_start
+                for i in range(q):
+
+                    dest_gt,_,_,_,x1_,x2_ = myenv_eval.initial_state()
+
+                    dest_obs = torch.cat([x1_,x2_], dim=3)
+                    dest_code = eval_net.encode((dest_obs*1.0).cuda())[0].item()
+
+                    source_gt,_,_,_,x1_,x2_ = myenv_eval.initial_state()
+
+                    source_obs = torch.cat([x1_,x2_], dim=3)
+                    source_code = eval_net.encode((source_obs*1.0).cuda())[0].item()
+
+                    dist_matrix, predecessors, _ =  dijkstra(min_only= True, csgraph=dijkstra_adjacency_matrix, directed=True, indices=source_code, return_predecessors=True)
+
+                    if dist_matrix[dest_code] == np.inf:
+                        continue
+                    curr = dest_code
+                    actions = []
+                    while(curr != source_code):
+                        prev = predecessors[curr]
+                        act = transitions_for_dijkstra[prev,curr]
+                        assert act != -1
+                        actions = [act] + actions
+                        curr = prev
+                    gt = source_gt
+                    for act in actions:
+                        gt,_,_,_,_,_ = myenv_eval.step(act, act.item())
+                    if (gt == dest_gt):
+                        wins += 1
+                torch.save({"total": + q, "wins": wins}, args.log_eval_prefix + "_it_"+ str(mybuffer.num_ex) + "_train_it_"+ str(iteration+1) + "_seed_" + str(args.seed) + ".pth")
+                print("Codes to GT:")
+                for i in range(args.ncodes):
+                    sum_ = codes_by_gt[i].sum()
+                    inds = np.flip(np.argsort(codes_by_gt[i]))[:4]
+                    inds_prob_1 = codes_by_gt[i,inds[0]]/sum_
+                    inds_prob_2 = codes_by_gt[i,inds[1]]/sum_
+                    inds_prob_3 = codes_by_gt[i,inds[2]]/sum_
+                    inds_prob_4 = codes_by_gt[i,inds[3]]/sum_
+                    print("Code " + str(i)+": freq: " + str(sum_) + " top GT: " + str(inds[0]) + " (" + str(inds_prob_1*100.) + "%), "+ str(inds[1]) + " (" + str(inds_prob_2*100.) + "%), "+ str(inds[2]) + " (" + str(inds_prob_3*100.) + "%), "+ str(inds[3]) + " (" + str(inds_prob_4*100.) + "%)")
+                print("GT to Codes:")
+                for i in range(myenv.total_states):
+                    sum_ = codes_by_gt[:,i].sum()
+                    inds = np.flip(np.argsort(codes_by_gt[:,i]))[:4]
+                    inds_prob_1 = codes_by_gt[inds[0],i]/sum_
+                    inds_prob_2 = codes_by_gt[inds[1],i]/sum_
+                    inds_prob_3 = codes_by_gt[inds[2],i]/sum_
+                    inds_prob_4 = codes_by_gt[inds[3],i]/sum_
+                    print("GT " + str(i)+": freq: " + str(sum_) + " top Codes: " + str(inds[0]) + " (" + str(inds_prob_1*100.) + "%), "+ str(inds[1]) + " (" + str(inds_prob_2*100.) + "%), "+ str(inds[2]) + " (" + str(inds_prob_3*100.) + "%), "+ str(inds[3]) + " (" + str(inds_prob_4*100.) + "%)")
+                print("Out of " + str(q) + " path-finding trials, suceeded in " + str(wins) + ".")
+            eval_net.train()
```
#### Diff for `encoders/mlp_enc1.py`
Code for the encoder. Minor change made to allow for new environment.
```diff
@@ -33 +33 @@
-            if self.args.exo_noise == "two_maze" and args.data == 'maze':
+            if self.args.exo_noise == "two_maze" and (args.data == 'maze' or args.data == 'periodic-cart') :
```
#### Diff for `encoders/mlp_pred1.py`
This file defines the multistep inverse model, as well as some code used for baseline methods in Lamb et al 2022. We add code for the latent forward model. (Note that there was a pre-existing commented-out line defining a a ``self.forward_mlp`` model in this file, but this was not used anywhere else in the code, so it is not clear if or how it was ever used to train the encoder.)
```diff
@@ -38,0 +39,2 @@
+        if (self.args.use_forward):
+            self.out_fwd = nn.Sequential(nn.Linear(512+10, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, ncodes)) # Following the above by assuming <= 10 actions

```
#### Diff for `transition.py`
File defines transition matrix. 
- Remove excessive printing
```diff
@@ -59 +59 @@
-            print('types', ind_last.device, y1.device)
+            #print('types', ind_last.device, y1.device)
```
- Bug fix needed to run
```diff
@@ -97 +97,2 @@
-                mode = mode[0][0]
+                #mode = mode[0][0]
+                mode = mode[0]
```
