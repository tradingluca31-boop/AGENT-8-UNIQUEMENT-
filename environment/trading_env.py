"""
Trading Environment V2.7 NUCLEAR - AGENT 8 MODE COLLAPSE FIX
================================================================================
INSTITUTIONAL-GRADE FIX FOR 100% HOLD MODE COLLAPSE

VERSION: 2.7 NUCLEAR
Date: 2025-11-25
Problem V2.6: Logit gap +1081 (100% HOLD, 0 trades)
Solution V2.7: 8 NUCLEAR FIXES

================================================================================
V2.7 NUCLEAR FIXES (vs V2.6 FAILED)
================================================================================

FIX 1: TRADING ACTION REWARDS (NEW - NUCLEAR) ✅ PROTECTED!
├─ Open trade (BUY/SELL): +5.0 (BOOSTED from +2.0, vs 0.0 in V2.6)
├─ Close profitable: +5.0 (BOOSTED from +2.0, vs +0.80 max in V2.6)
├─ Close loss: -1.0 (proportional to +5.0, vs -0.5 before)
├─ CRITICAL FIX: Added AFTER reward scaling (not diluted!)
└─ Result: Even losing trades are POSITIVE (+4.0 vs HOLD=0.0)

FIX 2: BONUSES × 20 (vs × 4 in V2.6)
├─ Direction prediction: 0.08 → 0.40 (×5)
├─ Profit taking 4R: 0.80 → 4.00 (×5)
├─ Profit taking 2R: 0.40 → 2.00 (×5)
├─ Loss cutting: 0.12 → 0.60 (×5)
└─ Trade completion: 0.40 → 2.00 (×5)

FIX 3: HOLD PENALTY EXPONENTIELLE (NEW)
├─ Threshold: 5 consecutive holds (vs 15 in V2.6)
├─ Formula: -2.0 × ((holds-5)/5)²
├─ 5 holds → 0.00
├─ 10 holds → -2.00
├─ 15 holds → -8.00 (vs -0.15 max in V2.6)
└─ 20 holds → -18.00 (CATASTROPHIC)

FIX 4: ACTION MASKING 5/10 (vs 8/10 in V2.6)
└─ Block action if repeated ≥5 times in last 10 (more aggressive)

FIX 5: DEMONSTRATION LEARNING (NEW - PHASES 1-3) ✅ WIDENED!
├─ Phase 1 (0-100K): Force smart trades (RSI <40/>60 - WIDENED!) + MEGA rewards (+10.0)
├─ Phase 2 (100K-300K): Reduce forcing (50%→0%), same thresholds, rewards (+5.0)
└─ Phase 3 (300K-500K): Autonomy, amplified rewards (+2.0)

FIX 6: FORCED TRADING (NEW - SAFETY)
└─ If 0 trades after 1000 steps → Force BUY or SELL (break paralysis)

FIX 7: FEATURE REMOVAL (SUSPECTS)
├─ xauusd_d1_volume_sma_20 (variance 18027 - ÉNORME)
├─ xauusd_h1_volume_sma_20 (variance 1674)
└─ Other high-variance volume features

FIX 8: OVER-TRADING PROTECTION (NEW)
└─ Max 1 trade per 10 bars (prevent reward hacking)

================================================================================
EXPECTED RESULTS V2.7 (vs V2.6 CATASTROPHE)
================================================================================

V2.6 Results:
  - P(HOLD) = 100.0% (mode collapse total)
  - Logit gap = +1081 (pire jamais vu)
  - Trades = 0 (paralysie complète)
  - Entropy = 0.0 (surconfiance)

V2.7 Expected:
  - P(HOLD) < 40% (diversité forcée)
  - Logit gap < 50 (convergence saine)
  - Trades > 50 per checkpoint (activité)
  - Entropy 0.20-0.40 (exploration maintenue)

Standards: Renaissance Technologies, Two Sigma, Citadel
Papers: Demonstration Learning (Hester et al. 2018), Entropy Regularization

STRATEGY: Mean Reversion Trading (M15 timeframe)
- Entry: Oversold/Overbought zones (RSI, BB, Stochastic)
- Exit: Fixed TP (4R) or Fixed SL (ATR-based)
- Hold Time: 15min-4h (quick reversions)

FEATURE ENGINEERING RL:
12. Recent action history (5 actions)
13. Regret signal (missed opportunities)
14. Position duration tracking
15. Unrealized PnL ratio
16. Market regime classification
17. Time until macro event
18. Volatility percentile
19. Trade Similarity Score (Pattern recognition vs winners/losers)

Total features: 209 base (199+10 temporal V3) + 13 RL = 222 features

SAC-SPECIFIC ADAPTATIONS:
- Continuous action space: Box([-1, 1]) - direction & confidence
- Automatic Entropy Tuning enabled (ent_coef='auto')
- Twin Q-Networks for stable learning
- No discrete action conversion needed

Based on research from: Renaissance Technologies, Citadel, Two Sigma
Papers: Haarnoja et al. (2018) - SAC, Schulman et al. (2017), Ng et al. (1999)

Author: Claude Code
Date: 2025-11-17
Version: V2.0 - Agent 8 Mean Reversion
"""

import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from collections import deque, Counter
from scipy import stats


class GoldTradingEnvAgent8(gym.Env):
    """
    Environnement Gymnasium pour trading Gold - AGENT 8 MEAN REVERSION (SAC)

    Observation space: Box(221,) - 209 base features (199+10 temporal V3) + 12 RL features
    Action space: Box([-1, 1]) - Continuous (SAC compatible)
        - action[0]: direction (-1=SELL, 0=HOLD, +1=BUY)

    Reward: Combinaison Sharpe + Profit + Drawdown penalty + Win rate

    Contraintes FTMO:
        - Max 1% risk per trade
        - Max 2% daily loss (stop si atteint)
        - Max 10% drawdown (termine épisode en production)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        features_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        initial_balance: float = 100000.0,
        max_episode_steps: int = 5000,
        verbose: bool = False,
        training_mode: bool = True
    ):
        """
        Args:
            features_df: DataFrame avec 209 features de base - 199+10 temporal V3 (index datetime)
            prices_df: DataFrame avec OHLCV XAUUSD M15 (pour exécution trades)
            initial_balance: Balance initiale ($100K default)
            max_episode_steps: Durée max épisode (5000 steps = ~52 jours en M15)
            verbose: Afficher logs détaillés
            training_mode: Si True, continue après 10% DD (training)
                          Si False, termine à 10% DD (backtest/live)
        """
        super().__init__()

        self.features_df = features_df
        self.prices_df = prices_df
        self.initial_balance = initial_balance
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose
        self.training_mode = training_mode

        # Aligner features et prices
        common_idx = self.features_df.index.intersection(self.prices_df.index)
        self.features_df = self.features_df.loc[common_idx]
        self.prices_df = self.prices_df.loc[common_idx]

        self.total_steps = len(self.features_df)

        # Observation space: 209 base (199+10 temporal V3) + 13 RL = 222 features
        n_base_features = self.features_df.shape[1]  # Should be 209 (199+10 temporal V3)
        n_rl_features = 13  # 13 RL-specific features (including trade_similarity_score)
        n_total_features = n_base_features + n_rl_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_total_features,),
            dtype=np.float32
        )

        # Action space: Discrete (PPO) - CONVERTED FROM SAC
        # 0 = SELL, 1 = HOLD, 2 = BUY
        self.action_space = spaces.Discrete(3)

        # Constants - FTMO & Trading (P2: FREE LEARNING MODE)
        # During training: Relaxed limits to allow exploration
        # During backtest/live: Strict FTMO limits
        if training_mode:
            self.FTMO_MAX_RISK_PER_TRADE = 0.01  # 1% (unchanged)
            self.FTMO_MAX_DAILY_LOSS = 0.20  # 20% (relaxed from 2%)
            self.FTMO_MAX_DRAWDOWN = 0.50  # 50% (relaxed from 10%)
        else:
            self.FTMO_MAX_RISK_PER_TRADE = 0.01  # 1%
            self.FTMO_MAX_DAILY_LOSS = 0.02  # 2% (FTMO strict)
            self.FTMO_MAX_DRAWDOWN = 0.10  # 10% (FTMO strict)
        self.RISK_REWARD_RATIO = 4.0  # 4R TP
        self.MIN_CONFIDENCE_THRESHOLD = 0.30  # Lower for SAC continuous
        self.SPREAD_PIPS = 0.5  # Gold spread
        self.SLIPPAGE_PIPS = 0.3
        self.COMMISSION_PER_LOT = 7.0
        self.XAUUSD_PIP_VALUE = 0.01
        self.XAUUSD_CONTRACT_SIZE = 100.0
        self.SHARPE_WINDOW = 252

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.equity = initial_balance
        self.position = 0.0
        self.position_side = 0  # -1 (short), 0 (flat), 1 (long)
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.entry_features = None  # WALL STREET: Store 15 features at entry (Trade Pattern Memory)

        # TP/SL Automatic (4R, ATR-based)
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.initial_risk_amount = 0.0

        # FTMO tracking
        self.daily_pnl = 0.0
        self.daily_start_balance = initial_balance
        self.last_date = None
        self.max_balance = initial_balance
        self.max_drawdown = 0.0
        self.daily_loss_limit_reached = False
        self.risk_multiplier = 1.0

        # Metrics
        self.trades = []
        self.daily_returns = deque(maxlen=self.SHARPE_WINDOW)
        self.ftmo_violations = 0

        # V2 ULTIMATE: Risk Management Metrics
        self.var_95 = 0.0  # Value at Risk 95% (5th percentile of daily returns)
        self.tail_risk_detected = False  # Kurtosis > 3.0 flag

        # Action tracking
        self.consecutive_holds = 0
        self.last_action = 0.0  # Continuous action
        self.total_holds = 0
        self.total_actions = 0
        self.recent_actions = deque(maxlen=100)

        # F2: EXPLICIT DIVERSITY PENALTY - Track last 1000 actions
        self.action_history_1000 = deque(maxlen=1000)

        # F3: ACTION MASKING - Track last 10 actions for masking
        self.last_10_actions = deque(maxlen=10)

        # F4: CRITIC VARIANCE BONUS - Track last 100 value estimates
        self.value_history = deque(maxlen=100)

        # V2: Curriculum Learning
        self.curriculum_level = 1
        self.last_price_direction = 0
        self.last_close_price = None

        # V2: RL-specific features
        self.action_history = deque(maxlen=5)
        self.regret_signal = 0.0
        self.position_entry_step = 0
        self.unrealized_pnl_ratio = 0.0
        self.market_regime = 0
        self.hours_until_event = 999.0
        self.volatility_percentile = 0.5

        # V2: Advanced Risk Management
        self.recent_returns = deque(maxlen=100)
        self.kelly_fraction = 0.0
        self.var_95 = 0.0
        self.tail_risk_detected = False

        # V2: Performance tracking
        self.performance_score = 0.0
        self.adaptive_reward_multiplier = 1.0

        # FIX 1 V2.7 NUCLEAR: Trading Action Reward Tracking
        # Track if position opened/closed THIS step for immediate rewards
        self.position_opened_this_step = False
        self.position_closed_this_step = False
        self.last_closed_pnl = 0.0  # To check if profit or loss

        # FIX 8 V2.7 NUCLEAR: Over-Trading Protection
        # Track last trade opening step to prevent reward hacking (max 1 trade per 10 bars)
        self.last_trade_open_step = -999  # Initialize to allow first trade

        # FIX 5 V2.7 NUCLEAR: Demonstration Learning
        # Track global timestep across all episodes (for phase detection)
        self.global_timestep = 0  # Will be updated by training loop via set_global_timestep()
        self.demonstration_trade_this_step = False  # Flag if trade was forced by demonstration

        # V2 ULTIMATE: Trade Pattern Memory (WALL STREET - Progressive Growth)
        # Stores 15 entry features for each trade
        # Compares with top N best + bottom N worst (N grows 10→15→20→25)
        self.trade_similarity_score = 0.0  # [-1, +1]
        self.memory_active = False  # Activates after 10 trades
        self.memory_capacity = 10  # Start at 10, grows to 25
        self.winner_patterns = []  # List of dict with winner trade patterns (best PnL%)
        self.loser_patterns = []   # List of dict with loser trade patterns (worst PnL%)

        self.log("[BUILD] Trading Environment V2 - AGENT 8 MEAN REVERSION (SAC)")
        self.log(f"   Observation space: {n_total_features} features ({n_base_features} base + {n_rl_features} RL)")
        self.log(f"   Action space: Continuous Box([-1, 1]) - SAC compatible")
        self.log(f"   TP/SL: 4R Take Profit, ATR-based Stop Loss")
        self.log(f"   Features: {self.features_df.shape}")
        self.log(f"   Prices: {self.prices_df.shape}")
        self.log(f"   Total steps: {self.total_steps}")
        self.log(f"   Initial balance: ${self.initial_balance:,.2f}")

    def log(self, message: str):
        """Log si verbose"""
        if self.verbose:
            print(message)

    def set_curriculum_level(self, level: int):
        """Set curriculum difficulty level (1-4)"""
        if level < 1 or level > 4:
            raise ValueError("Curriculum level must be between 1 and 4")
        self.curriculum_level = level
        self.log(f"[CURRICULUM] Level set to {level}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)

        # Random start
        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            max_start = self.total_steps - self.max_episode_steps - 1
            self.current_step = np.random.randint(0, max(1, max_start))

        # Reset all state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0

        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.initial_risk_amount = 0.0
        self.position_duration_bars = 0  # FIX V2.1: Track position age

        self.daily_pnl = 0.0
        self.daily_start_balance = self.initial_balance
        self.last_date = None
        self.max_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.daily_loss_limit_reached = False
        self.risk_multiplier = 1.0

        self.trades = []
        self.daily_returns.clear()
        self.ftmo_violations = 0

        self.consecutive_holds = 0
        self.last_action = 0.0
        self.total_holds = 0
        self.total_actions = 0
        self.recent_actions.clear()

        # F2, F3, F4: Clear tracking buffers
        self.action_history_1000.clear()
        self.last_10_actions.clear()
        self.value_history.clear()

        self.last_price_direction = 0
        self.last_close_price = self.prices_df['close'].iloc[self.current_step]

        self.action_history.clear()
        self.regret_signal = 0.0
        self.position_entry_step = 0
        self.unrealized_pnl_ratio = 0.0
        self.market_regime = 0
        self.hours_until_event = 999.0
        self.volatility_percentile = 0.5

        self.recent_returns.clear()
        self.kelly_fraction = 0.0
        self.var_95 = 0.0
        self.tail_risk_detected = False

        self.performance_score = 0.0
        self.adaptive_reward_multiplier = 1.0

        # V2 ULTIMATE: Trade Pattern Memory reset
        self.trade_similarity_score = 0.0
        # NOTE: We DO NOT reset winner_patterns and loser_patterns
        # They persist across episodes (accumulate learning)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def set_global_timestep(self, timestep: int):
        """
        FIX 5 V2.7 NUCLEAR: Set global timestep for Demonstration Learning phases
        Called by training loop (e.g., after each model.learn() call)
        """
        self.global_timestep = timestep

    def _get_demonstration_phase(self) -> int:
        """
        FIX 5 V2.7 NUCLEAR: Get current demonstration learning phase

        Returns:
            0 = No demonstration (>300K)
            1 = Phase 1 (0-100K): Force 100% smart trades + MEGA rewards (+10.0)
            2 = Phase 2 (100K-300K): Force 50%→0% + amplified rewards (+5.0)
            3 = Phase 3 (300K-500K): Autonomy + amplified rewards (+2.0)
        """
        if self.global_timestep < 100000:
            return 1  # Phase 1: Maximum demonstration
        elif self.global_timestep < 300000:
            return 2  # Phase 2: Reducing demonstration
        elif self.global_timestep < 500000:
            return 3  # Phase 3: Autonomy with bonuses
        else:
            return 0  # Normal operation

    def _should_force_demonstration_trade(self, current_phase: int, current_price: float) -> int:
        """
        FIX 5 V2.7 NUCLEAR: Check if should force a demonstration trade

        Args:
            current_phase: 1, 2, or 3
            current_price: Current market price

        Returns:
            0 = No force (let agent decide)
            1 = Force BUY (smart long opportunity)
            2 = Force SELL (smart short opportunity)
        """
        if current_phase == 0 or self.position_side != 0:
            return 0  # No forcing in Phase 0 or if already in position

        # Get RSI for Mean Reversion signal (Agent 8 specialty)
        try:
            rsi_col = 'rsi_14_m15' if 'rsi_14_m15' in self.features_df.columns else 'rsi_14'
            rsi = self.features_df[rsi_col].iloc[self.current_step]
        except (KeyError, IndexError):
            return 0  # No RSI available

        # Mean Reversion logic (WIDENED THRESHOLDS for more opportunities):
        # RSI < 40 = Oversold → BUY (expect reversion up)
        # RSI > 60 = Overbought → SELL (expect reversion down)

        if current_phase == 1:  # Phase 1 (0-100K): Force 100% of smart opportunities
            if rsi < 40:  # WIDENED from 30 to 40
                return 1  # Force BUY
            elif rsi > 60:  # WIDENED from 70 to 60
                return 2  # Force SELL
        elif current_phase == 2:  # Phase 2 (100K-300K): Force 50% → 0%
            # Linear decay: 100K=50%, 200K=25%, 300K=0%
            progress = (self.global_timestep - 100000) / 200000  # 0.0 to 1.0
            force_probability = 0.5 * (1.0 - progress)  # 50% → 0%

            if np.random.rand() < force_probability:
                if rsi < 40:  # WIDENED from 30 to 40
                    return 1  # Force BUY
                elif rsi > 60:  # WIDENED from 70 to 60
                    return 2  # Force SELL

        # Phase 3 or no smart opportunity
        return 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step

        Args:
            action: Continuous action from SAC
                action[0] = direction (-1 à +1)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # FIX 1 V2.7 NUCLEAR: Reset trading action flags at start of step
        self.position_opened_this_step = False
        self.position_closed_this_step = False
        self.last_closed_pnl = 0.0

        # Parse discrete action (PPO) - SIMPLIFIED FROM SAC
        # action is already discrete: 0=SELL, 1=HOLD, 2=BUY
        action_discrete = int(action)

        # =====================================================================
        # FIX 4 V2.7 NUCLEAR: ACTION MASKING 5/10 (MORE AGGRESSIVE)
        # =====================================================================
        # If action repeated ≥5/10 times → BLOCK IT (force other actions)
        # V2.6 was 8/10 (too lenient) → V2.7 is 5/10 (more aggressive)
        # This PREVENTS mode collapse by forcing exploration
        # Standard: Citadel, Two Sigma (constrained exploration)
        if len(self.last_10_actions) >= 10:
            # Count occurrences of each action in last 10
            action_counts = Counter(self.last_10_actions)

            # FIX 4: Changed from ≥8 to ≥5 (more aggressive masking)
            if action_counts.get(action_discrete, 0) >= 5:
                # Force a DIFFERENT action (random from the other 2)
                available_actions = [a for a in [0, 1, 2] if a != action_discrete]
                action_discrete = np.random.choice(available_actions)

                if self.verbose:
                    self.log(f"   [ACTION MASK V2.7] Action {action} blocked (repeated ≥5/10) → Forced to {action_discrete}")

        # Confidence (for PPO, all actions have equal confidence)
        confidence = 1.0 if action_discrete != 1 else 0.5  # Higher for BUY/SELL

        # =====================================================================
        # FIX 5 V2.7 NUCLEAR: DEMONSTRATION LEARNING (FORCE SMART TRADES)
        # =====================================================================
        # Phase 1 (0-100K): Force 100% smart trades (RSI oversold/overbought)
        # Phase 2 (100K-300K): Force 50%→0% smart trades
        # Phase 3 (300K-500K): Autonomy + amplified rewards
        current_phase = self._get_demonstration_phase()
        current_price = self.prices_df['close'].iloc[self.current_step]

        forced_action = self._should_force_demonstration_trade(current_phase, current_price)
        if forced_action > 0:
            # Override agent's action with demonstration trade
            original_action = action_discrete
            if forced_action == 1:  # Force BUY
                action_discrete = 2
            elif forced_action == 2:  # Force SELL
                action_discrete = 0

            self.demonstration_trade_this_step = True
            if self.verbose:
                self.log(f"   [FIX 5 V2.7] DEMONSTRATION PHASE {current_phase}: Forced {['SELL', '', 'BUY'][action_discrete+1]} (was {['SELL', 'HOLD', 'BUY'][original_action]})")
        else:
            self.demonstration_trade_this_step = False

        # Track action (F2, F3)
        self.total_actions += 1
        self.recent_actions.append(action_discrete)
        self.action_history.append(action_discrete)
        self.action_history_1000.append(action_discrete)  # F2: Track for diversity penalty
        self.last_10_actions.append(action_discrete)      # F3: Track for action masking

        # Update state (check TP/SL, daily reset, etc.)
        self._update_state()

        # =====================================================================
        # FIX 6 V2.7 NUCLEAR: FORCED TRADING (Safety Net for Paralysis)
        # =====================================================================
        # If agent refuses to trade after 1000 steps → FORCE BUY or SELL
        # This breaks paralysis loop and forces learning
        if self.current_step > 1000 and len(self.trades) == 0 and self.position_side == 0:
            if action_discrete == 1:  # Agent wants HOLD but stuck
                # Force a random trade (BUY or SELL)
                action_discrete = np.random.choice([0, 2])  # SELL or BUY
                if self.verbose:
                    self.log(f"   [FIX 6 V2.7] FORCED TRADING: 0 trades after {self.current_step} steps → Forcing action {action_discrete}")

        # Execute action
        current_price = self.prices_df['close'].iloc[self.current_step]

        if action_discrete == 1:  # HOLD
            self.consecutive_holds += 1
            self.total_holds += 1
        else:
            self.consecutive_holds = 0

            # Check if can trade
            if not self.daily_loss_limit_reached:
                if action_discrete == 0:  # SELL
                    if self.position_side == 0:  # Open SHORT
                        self._open_position(current_price, direction=-1, confidence=confidence)
                    elif self.position_side == 1:  # Close LONG
                        self._close_position(current_price)

                elif action_discrete == 2:  # BUY
                    if self.position_side == 0:  # Open LONG
                        self._open_position(current_price, direction=1, confidence=confidence)
                    elif self.position_side == -1:  # Close SHORT
                        self._close_position(current_price)

        self.last_action = action_discrete  # PPO: discrete action (0, 1, 2)

        # Update regret signal (RL feature)
        self.regret_signal = self._calculate_regret()

        # Update market regime classification (RL feature)
        self.market_regime = self._classify_market_regime()

        # Update hours until macro event (RL feature)
        self.hours_until_event = self._hours_until_macro_event()

        # Update volatility percentile (RL feature)
        self.volatility_percentile = self._volatility_percentile()

        # PHASE 2: Update adaptive reward multiplier (Curriculum Learning)
        # Update every 100 steps to avoid overhead
        if self.total_actions % 100 == 0:
            self._update_adaptive_reward_multiplier()

        # Move to next step
        self.current_step += 1

        # Check if episode done
        terminated = False
        truncated = False

        if self.current_step >= self.total_steps - 1:
            truncated = True

        if self.current_step >= self.max_episode_steps:
            truncated = True

        # Check FTMO max DD (only in production mode)
        if not self.training_mode:
            if self.max_drawdown >= self.FTMO_MAX_DRAWDOWN:
                terminated = True
                self.log(f"   [FTMO] MAX DD HIT: {self.max_drawdown:.2%}")

        # Calculate reward
        reward = self._calculate_reward()

        # Get observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _open_position(self, current_price: float, direction: int, confidence: float):
        """Open a new position (LONG or SHORT)"""
        if self.position_side != 0:
            return  # Already in position

        # FIX 8 V2.7 NUCLEAR: Over-Trading Protection (max 1 trade per 10 bars)
        # Prevents reward hacking by opening/closing rapidly
        if self.current_step - self.last_trade_open_step < 10:
            if self.verbose:
                bars_since = self.current_step - self.last_trade_open_step
                self.log(f"   [FIX 8 V2.7] Over-trading blocked: Only {bars_since} bars since last trade (min 10)")
            return  # Block trade

        # Get ATR for SL calculation
        atr = self._get_atr()

        # V2 ULTIMATE: Store 15 entry features for Trade Pattern Memory (WALL STREET)
        # Agent 8 M15 features (Mean Reversion specific)
        try:
            self.entry_features = np.array([
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('rsi_14_m15' if 'rsi_14_m15' in self.features_df.columns else 'rsi_14')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd_m15' if 'macd_m15' in self.features_df.columns else 'macd')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('macd_signal_m15' if 'macd_signal_m15' in self.features_df.columns else 'macd_signal')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('atr_14_m15' if 'atr_14_m15' in self.features_df.columns else 'atr_14')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('adx_m15' if 'adx_m15' in self.features_df.columns else 'adx')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('stochastic_k_m15' if 'stochastic_k_m15' in self.features_df.columns else 'stochastic_k')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('cci_m15' if 'cci_m15' in self.features_df.columns else 'cci')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('williams_r_m15' if 'williams_r_m15' in self.features_df.columns else 'williams_r')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('roc_m15' if 'roc_m15' in self.features_df.columns else 'roc')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('mfi_m15' if 'mfi_m15' in self.features_df.columns else 'mfi')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('bb_width_m15' if 'bb_width_m15' in self.features_df.columns else 'bb_width')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('volume_ratio_m15' if 'volume_ratio_m15' in self.features_df.columns else 'volume_ratio')],
                self.features_df.iloc[self.current_step, self.features_df.columns.get_loc('trend_strength_m15' if 'trend_strength_m15' in self.features_df.columns else 'trend_strength')],
                float(self.volatility_percentile),  # Calculated feature
                float(self.max_drawdown)  # Current DD at entry
            ], dtype=np.float32)
        except (KeyError, IndexError):
            # If features not available, default vector
            self.entry_features = np.zeros(15, dtype=np.float32)

        # V2 ULTIMATE: Calculate trade similarity score (25+25 cosine similarity)
        trade_similarity = self._calculate_trade_similarity_score()
        self.trade_similarity_score = trade_similarity

        # Adjust confidence based on trade similarity
        # High similarity to winners (+0.8) → Boost confidence
        # High similarity to losers (-0.8) → Reduce confidence
        # SOFTENED: 0.15 instead of 0.3 to avoid blocking trades (±12% instead of ±24%)
        sim_multiplier = 1.0 + (trade_similarity * 0.15)  # Range: [0.88, 1.12]
        adjusted_confidence = confidence * sim_multiplier
        adjusted_confidence = np.clip(adjusted_confidence, 0.3, 1.0)

        self.log(f"   [SIMILARITY] Score: {trade_similarity:+.2f} | Confidence: {confidence:.2f} -> {adjusted_confidence:.2f}")

        # Calculate position size
        base_risk = self._calculate_adaptive_base_risk(adjusted_confidence, atr)
        risk_amount = self.balance * base_risk * self.risk_multiplier

        # ATR-based stop distance
        stop_distance = atr * 1.0  # 1.0x ATR - Mean Reversion M15 (optimal TP/SL)

        # Calculate position size (lots)
        price_risk_per_lot = stop_distance * self.XAUUSD_CONTRACT_SIZE * self.XAUUSD_PIP_VALUE
        if price_risk_per_lot > 0:
            position_lots = risk_amount / price_risk_per_lot
            position_lots = max(0.01, min(position_lots, 10.0))  # Clip 0.01 to 10 lots
        else:
            position_lots = 0.01

        # Open position
        self.position = position_lots
        self.position_side = direction
        self.entry_price = current_price
        self.initial_risk_amount = risk_amount
        self.position_entry_step = self.current_step
        self.position_duration_bars = 0  # Track position age (M15 bars)

        # Set SL/TP (4R)
        if self.position_side == 1:  # LONG
            self.stop_loss_price = current_price - stop_distance
            self.take_profit_price = current_price + (stop_distance * self.RISK_REWARD_RATIO)
        else:  # SHORT
            self.stop_loss_price = current_price + stop_distance
            self.take_profit_price = current_price - (stop_distance * self.RISK_REWARD_RATIO)

        # Transaction costs
        spread_cost = self.SPREAD_PIPS * self.XAUUSD_PIP_VALUE * self.XAUUSD_CONTRACT_SIZE * position_lots
        slippage_cost = self.SLIPPAGE_PIPS * self.XAUUSD_PIP_VALUE * self.XAUUSD_CONTRACT_SIZE * position_lots
        total_cost = spread_cost + slippage_cost + self.COMMISSION_PER_LOT * position_lots

        self.balance -= total_cost

        # FIX 1 V2.7 NUCLEAR: Mark position opened this step (for +2.0 reward)
        self.position_opened_this_step = True

        # FIX 8 V2.7 NUCLEAR: Update last trade open step (for over-trading protection)
        self.last_trade_open_step = self.current_step

        self.log(f"   [OPEN] {['SHORT', '', 'LONG'][self.position_side+1]} "
                f"{position_lots:.2f} lots @ {current_price:.2f}")
        self.log(f"      Risk: ${risk_amount:.2f} ({base_risk*100:.2f}%)")
        self.log(f"      SL: {self.stop_loss_price:.2f} | TP: {self.take_profit_price:.2f} (4R)")

    def _close_position(self, current_price: float):
        """Close current position"""
        if self.position_side == 0:
            return

        # Calculate PnL
        price_diff = (current_price - self.entry_price) * self.position_side
        pnl = price_diff * self.XAUUSD_CONTRACT_SIZE * self.position

        # Costs
        spread_cost = self.SPREAD_PIPS * self.XAUUSD_PIP_VALUE * self.XAUUSD_CONTRACT_SIZE * self.position
        slippage_cost = self.SLIPPAGE_PIPS * self.XAUUSD_PIP_VALUE * self.XAUUSD_CONTRACT_SIZE * self.position
        total_cost = spread_cost + slippage_cost + self.COMMISSION_PER_LOT * self.position

        net_pnl = pnl - total_cost

        # Update balance
        self.balance += net_pnl
        self.daily_pnl += net_pnl

        # Store trade
        pnl_pct = (net_pnl / self.initial_risk_amount) if self.initial_risk_amount > 0 else 0.0
        trade_dict = {
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'entry_step': self.position_entry_step,
            'exit_step': self.current_step,
            'position_side': self.position_side,
            'position_size': self.position,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'hold_duration': self.current_step - self.position_entry_step
        }
        self.trades.append(trade_dict)

        # Store pattern for win probability learning (V2 ULTIMATE)
        self._store_trade_pattern(trade_dict)

        # FIX 1 V2.7 NUCLEAR: Mark position closed this step (for +2.0 or -0.5 reward)
        self.position_closed_this_step = True
        self.last_closed_pnl = net_pnl

        self.log(f"   [CLOSE] {['SHORT', '', 'LONG'][self.position_side+1]} "
                f"@ {current_price:.2f} | PnL: ${net_pnl:+.2f} ({pnl_pct:+.2%})")

        # Reset position
        self.position = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.initial_risk_amount = 0.0

    def _update_state(self):
        """Update state (unrealized PnL, check TP/SL, daily reset, etc.)"""
        current_price = self.prices_df['close'].iloc[self.current_step]
        current_date = self.features_df.index[self.current_step].date()

        # Unrealized PnL
        if self.position_side != 0:
            # Increment position duration
            if hasattr(self, 'position_duration_bars'):
                self.position_duration_bars += 1
            else:
                self.position_duration_bars = 0  # Initialize if not exists

            price_diff = (current_price - self.entry_price) * self.position_side
            self.unrealized_pnl = price_diff * self.XAUUSD_CONTRACT_SIZE * self.position

            # Check TP/SL
            sl_hit = False
            tp_hit = False

            if self.position_side == 1:  # LONG
                if current_price <= self.stop_loss_price:
                    sl_hit = True
                elif current_price >= self.take_profit_price:
                    tp_hit = True
            else:  # SHORT
                if current_price >= self.stop_loss_price:
                    sl_hit = True
                elif current_price <= self.take_profit_price:
                    tp_hit = True

            # FIX V2.1: Force close après 7 jours (672 bars M15 = 7 days)
            max_hold_bars = 672  # 7 days × 24h × 4 bars/hour
            if self.position_duration_bars >= max_hold_bars:
                self.log(f"   [MAX HOLD] Position held for {self.position_duration_bars} bars (7 days) - FORCE CLOSE @ {current_price:.2f}")
                self._close_position(current_price)
            elif sl_hit:
                self.log(f"   [SL] STOP LOSS HIT @ {current_price:.2f}")
                self._close_position(current_price)
            elif tp_hit:
                self.log(f"   [TP] TAKE PROFIT HIT (4R) @ {current_price:.2f}")
                self._close_position(current_price)
        else:
            self.unrealized_pnl = 0.0
            self.position_duration_bars = 0  # Reset when no position

        # Update equity
        self.equity = self.balance + self.unrealized_pnl

        # Daily reset
        if self.last_date is None or current_date > self.last_date:
            if self.last_date is not None:
                # Store daily return
                daily_return = (self.balance - self.daily_start_balance) / self.daily_start_balance
                self.daily_returns.append(daily_return)
                self.recent_returns.append(daily_return)

                # V2 ULTIMATE: Update risk metrics
                self.var_95 = self._calculate_var_95()  # VaR 95% (institutional)
                self.tail_risk_detected = self._calculate_tail_risk()  # Tail risk (kurtosis)

            # Reset daily tracking
            self.daily_pnl = 0.0
            self.daily_start_balance = self.balance
            self.daily_loss_limit_reached = False
            self.last_date = current_date

        # Check daily loss limit
        daily_loss_pct = self.daily_pnl / self.daily_start_balance if self.daily_start_balance > 0 else 0.0
        if daily_loss_pct <= -self.FTMO_MAX_DAILY_LOSS:
            self.daily_loss_limit_reached = True
            self.ftmo_violations += 1
            self.log(f"   [WARNING] DAILY LOSS LIMIT HIT: {daily_loss_pct:.2%}")

        # Update max balance and drawdown
        if self.equity > self.max_balance:
            self.max_balance = self.equity

        current_dd = (self.max_balance - self.equity) / self.max_balance if self.max_balance > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, current_dd)

        # Update risk multiplier (only in production)
        if not self.training_mode:
            self.risk_multiplier = self._calculate_risk_multiplier()

    def _calculate_reward(self) -> float:
        """Calculate reward (V2 Tiered Hierarchy + V2.7 NUCLEAR TRADING REWARDS)"""
        reward = 0.0

        # =====================================================================
        # FIX 1 V2.7 NUCLEAR: TRADING ACTION REWARDS (THE CORE FIX!) ✅ PROTECTED!
        # =====================================================================
        # Show agent that trading is INHERENTLY POSITIVE (not catastrophic)
        # BOOSTED from +2.0 to +5.0 and PROTECTED (added at END after scaling)
        # Even LOSING trades are positive (+4.0) vs HOLD (0.0)
        # Profitable trade: +5.0 (open) + 5.0 (close) = +10.0 ✅
        # Losing trade:     +5.0 (open) - 1.0 (close) = +4.0 ✅
        # HOLD:             0.0
        trading_action_reward = 0.0

        if self.position_opened_this_step:
            # Opening a trade = +5.0 (BOOSTED - immediate positive signal)
            trading_action_reward += 5.0
            if self.verbose:
                self.log(f"   [FIX 1 V2.7] Position opened → +5.0 reward")

        if self.position_closed_this_step:
            if self.last_closed_pnl > 0:
                # Profitable close = +5.0 (BOOSTED - double positive!)
                trading_action_reward += 5.0
                if self.verbose:
                    self.log(f"   [FIX 1 V2.7] Profitable close (${self.last_closed_pnl:+.2f}) → +5.0 reward")
            else:
                # Losing close = -1.0 (still positive overall when combined with +5.0 open)
                trading_action_reward -= 1.0
                if self.verbose:
                    self.log(f"   [FIX 1 V2.7] Losing close (${self.last_closed_pnl:+.2f}) → -1.0 reward")

        # NOTE: trading_action_reward is NOT added here (will be added at END after scaling)

        # =====================================================================
        # FIX 5 V2.7 NUCLEAR: DEMONSTRATION LEARNING MEGA BONUSES
        # =====================================================================
        # Give MASSIVE rewards for demonstration trades (show agent trading works!)
        # Phase 1 (0-100K): +10.0 bonus
        # Phase 2 (100K-300K): +5.0 bonus
        # Phase 3 (300K-500K): +2.0 bonus
        demonstration_bonus = 0.0

        if self.demonstration_trade_this_step:
            current_phase = self._get_demonstration_phase()
            if current_phase == 1:
                demonstration_bonus = 10.0  # MEGA bonus Phase 1
            elif current_phase == 2:
                demonstration_bonus = 5.0   # Large bonus Phase 2
            elif current_phase == 3:
                demonstration_bonus = 2.0   # Moderate bonus Phase 3

            if demonstration_bonus > 0 and self.verbose:
                self.log(f"   [FIX 5 V2.7] Demonstration trade bonus: +{demonstration_bonus:.1f}")

        reward += demonstration_bonus

        # FIX V2.1: Reward intermédiaire pour diversité (même sans trade fermé)
        # Encourage exploration et empêche mode collapse
        # NOTE: DO NOT return early! Must continue to add trading_action_reward at the end!
        if len(self.trades) < 1:
            # Reward basé sur diversité d'actions récentes
            if len(self.recent_actions) >= 5:
                action_diversity = len(set(self.recent_actions)) / 3.0  # 0.33 (mono) to 1.0 (balanced)
                diversity_reward = action_diversity * 0.3  # 0.10 to 0.30 (×3 stronger)

                # Bonus si agent alterne vraiment (pas juste 100% SELL)
                action_counts = Counter(self.recent_actions)
                max_action_pct = max(action_counts.values()) / len(self.recent_actions)

                if max_action_pct < 0.6:  # <60% single action = good diversity
                    diversity_reward += 0.15  # Bonus (×3 stronger)

                reward += diversity_reward  # FIX: Accumulate instead of early return!

        # FIX V2.4: PASSIVITY PENALTY (100% HOLD)
        # Si agent refuse de trader après 100 steps → PENALTY MASSIVE
        if self.current_step > 100 and len(self.trades) == 0:
            if len(self.recent_actions) >= 20:
                hold_count = sum(1 for a in list(self.recent_actions)[-20:] if a == 0)
                hold_pct = hold_count / 20.0

                if hold_pct > 0.75:  # >75% HOLD et 0 trades
                    passivity_penalty = -5.0  # TRÈS FORTE
                    reward += passivity_penalty  # FIX: Accumulate instead of early return!

        # FIX V2.3: NO 5 CONSECUTIVE SAME ACTIONS
        # Empêche répétitions mais trop faible
        if len(self.recent_actions) >= 5:
            last_5 = list(self.recent_actions)[-5:]

            # Si 5 actions consécutives identiques → penalty
            if len(set(last_5)) == 1:  # All same action
                repetition_penalty = -0.50  # Augmenté de 0.30 → 0.50
                reward += repetition_penalty  # FIX: Accumulate instead of early return!

        # Progressive reward scaling based on trade count
        if len(self.trades) == 1:
            reward_scale = 0.3  # 30% of full reward (learning phase)
        elif len(self.trades) == 2:
            reward_scale = 0.6  # 60% of full reward (growing confidence)
        else:  # 3+ trades
            reward_scale = 1.0  # 100% full reward (mature trading)

        # TIER 1: Core Performance (70%)
        # 1.1 Profit (40%) - AMPLIFIED REWARD FOR WINNERS
        total_pnl = sum(t['pnl'] for t in self.trades)
        roi = total_pnl / self.initial_balance if self.initial_balance > 0 else 0.0

        # FIX: MASSIVE amplification - Winners get ×15, Losers get ×5 (3× difference!)
        # This creates STRONG signal: "Winning is WAY better than losing!"
        if total_pnl > 0:  # WINNING
            profit_reward = roi * 15.0  # ×15 amplification for winners! [ROCKET]
        else:  # LOSING
            profit_reward = roi * 5.0   # ×5 penalty for losers (learn to avoid losses!)

        # 1.2 Sharpe Ratio (20%)
        sharpe_reward = 0.0
        if len(self.daily_returns) >= 30:
            returns_arr = np.array(list(self.daily_returns))
            if returns_arr.std() > 0:
                sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(252)
                sharpe_reward = sharpe * 0.5

        # 1.3 Drawdown penalty (10%)
        dd_penalty = -self.max_drawdown * 5.0

        core_performance = (profit_reward * 0.4 + sharpe_reward * 0.2 + dd_penalty * 0.1)

        # =====================================================================
        # TIER 2: RISK MANAGEMENT (20%) - V2 ULTIMATE INSTITUTIONAL
        # =====================================================================
        risk_score = 0.0

        # 2.1 FTMO Compliance (10% weight)
        ftmo_score = 0.0
        if self.max_drawdown < 0.05:  # Excellent < 5%
            ftmo_score += 0.1
        elif self.max_drawdown < 0.08:  # Good < 8%
            ftmo_score += 0.05
        if not self.daily_loss_limit_reached:  # No daily violations
            ftmo_score += 0.1

        # 2.2 VaR 95% Monitoring (5% weight)
        var_score = 0.0
        if len(self.daily_returns) >= 30:
            # VaR is NEGATIVE (e.g., -1.5%)
            # Good: VaR > -2% (i.e., loss < 2% in 95% cases)
            # Warning: VaR < -2.5%
            # Red Flag: VaR < -3%
            if self.var_95 > -0.015:  # Excellent (VaR > -1.5%)
                var_score = 0.1
            elif self.var_95 > -0.02:  # Good (VaR > -2%)
                var_score = 0.05
            elif self.var_95 < -0.03:  # RED FLAG (VaR < -3%)
                var_score = -0.2  # Penalty

        # 2.3 Tail Risk Detection (5% weight)
        tail_score = 0.0
        if len(self.daily_returns) >= 30:
            if self.tail_risk_detected:
                # Fat tails detected (kurtosis > 4.0) → BLACK SWAN RISK
                tail_score = -0.15  # Penalty for excessive tail risk
            else:
                # Normal distribution or thin tails → GOOD
                tail_score = 0.05

        # Combine risk scores
        risk_score = ftmo_score + var_score + tail_score

        # TIER 3: Behavioral (20% - INCREASED from 10% to fight mode collapse)
        # Diversity score
        diversity_score = self._calculate_diversity_score()

        # Combine (diversity weight DOUBLED to 20%)
        reward = (core_performance * 0.6 + risk_score * 0.2 + diversity_score * 0.2)

        # Adaptive scaling
        reward *= self.adaptive_reward_multiplier

        # =====================================================================
        # FIX 2 V2.7 NUCLEAR: BONUSES × 20 (was × 4 in V2.6)
        # V2.6 bonuses were ×4 but too weak vs +1081 logit bias
        # V2.7 bonuses are ×20 (multiply V2.6 values by 5) to DOMINATE logits
        # =====================================================================
        v2_bonuses = 0.0

        # Bonus 1: Direction Prediction Reward (+0.40 - was 0.08, ×5)
        # Reward even without trade if direction correct (encourages market reading)
        if self.position_side == 0 and self.last_close_price is not None:
            current_price = self.prices_df['close'].iloc[self.current_step]
            price_direction = 1 if current_price > self.last_close_price else -1
            # Check if our last action predicted correctly
            if self.last_action * price_direction > 0:
                v2_bonuses += 0.40  # FIX 2: ×20 total (0.08 × 5 = 0.40)

        # Bonus 2: Enhanced Profit Taking (progressive - ×20 all tiers)
        if len(self.trades) > 0:
            last_trade = self.trades[-1]
            pnl_r = last_trade['pnl_pct']  # PnL as % of initial risk
            if pnl_r >= 1.0:  # 4R (100% of risk) = EXCELLENT
                v2_bonuses += 4.00  # FIX 2: ×20 (0.80 × 5 = 4.00)
            elif pnl_r >= 0.5:  # 2R (50% of risk) = GOOD
                v2_bonuses += 2.00  # FIX 2: ×20 (0.40 × 5 = 2.00)
            elif pnl_r >= 0.25:  # 1R (25% of risk) = OK
                v2_bonuses += 1.00  # FIX 2: ×20 (0.20 × 5 = 1.00)
            elif pnl_r > 0:  # Any profit
                v2_bonuses += 0.40  # FIX 2: ×20 (0.08 × 5 = 0.40)

        # Bonus 3: Quick Loss Cutting (+0.60 - was 0.12, ×5)
        # Mean Reversion needs fast stop-loss on failed reversions
        if len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade['pnl'] < 0 and last_trade['pnl_pct'] > -0.5:  # Loss < 0.5R
                v2_bonuses += 0.60  # FIX 2: ×20 (0.12 × 5 = 0.60)

        # Bonus 4: Trade Completion Reward (+2.00 - was 0.40, ×5)
        # Encourages closing positions (prevents "forgotten" positions)
        if len(self.trades) > 0:
            # Check if this is a NEW trade (not counted yet)
            if not hasattr(self, '_last_trade_count'):
                self._last_trade_count = 0
            if len(self.trades) > self._last_trade_count:
                v2_bonuses += 2.00  # FIX 2: ×20 (0.40 × 5 = 2.00)
                self._last_trade_count = len(self.trades)

        # Add V2 bonuses to reward
        reward += v2_bonuses

        # =====================================================================
        # F2: EXPLICIT DIVERSITY PENALTY (Combat mode collapse)
        # =====================================================================
        # If >80% of last 1000 actions are the same → STRONG PENALTY
        # Standard: Renaissance Technologies (behavioral diversity enforcement)
        diversity_penalty = 0.0
        if len(self.action_history_1000) >= 100:  # Need at least 100 actions
            action_counts_1000 = Counter(self.action_history_1000)
            total_actions_1000 = len(self.action_history_1000)
            max_action_pct = max(action_counts_1000.values()) / total_actions_1000

            if max_action_pct > 0.80:  # >80% single action
                diversity_penalty = -2.0  # STRONG penalty
                if self.verbose and self.total_actions % 500 == 0:
                    most_common = action_counts_1000.most_common(1)[0]
                    self.log(f"   [DIVERSITY PENALTY] Action {most_common[0]} = {max_action_pct:.1%} of last 1000 → Penalty -2.0")

        reward += diversity_penalty

        # =====================================================================
        # P4: UNIFIED DIVERSITY SCORE (Shannon Entropy - Institutional Grade)
        # =====================================================================
        # Calculate Shannon entropy over last 1000 actions
        # Target: > 0.7 (70% of max entropy = log(3) = 1.0986)
        # Penalty if diversity < 0.5 (50% of max entropy)
        # Standard: Renaissance Technologies (entropy-based diversity)
        unified_diversity_penalty = 0.0
        if len(self.action_history_1000) >= 100:
            action_probs = []
            action_counts_1000 = Counter(self.action_history_1000)
            total = len(self.action_history_1000)

            for action in [0, 1, 2]:  # SELL, HOLD, BUY
                prob = action_counts_1000.get(action, 0) / total
                action_probs.append(prob)

            # Shannon entropy: H = -Σ(p * log(p))
            shannon_entropy = 0.0
            for p in action_probs:
                if p > 0:
                    shannon_entropy -= p * np.log(p)

            # Normalize (max entropy for 3 actions = log(3) = 1.0986)
            max_entropy = np.log(3.0)
            diversity_score = shannon_entropy / max_entropy  # 0 to 1

            # Penalty if diversity < 0.5 (50% of max)
            if diversity_score < 0.5:
                unified_diversity_penalty = -2.0
                if self.verbose and self.total_actions % 500 == 0:
                    self.log(f"   [UNIFIED DIVERSITY] Score: {diversity_score:.2f} (target >0.7) → Penalty -2.0")

        reward += unified_diversity_penalty

        # =====================================================================
        # FIX 3 V2.7 NUCLEAR: HOLD PENALTY EXPONENTIELLE (MASSIVE)
        # =====================================================================
        # V2.6: Start at 15 holds, max -0.15 (TOO WEAK)
        # V2.7: Start at 5 holds, max -18.0 (NUCLEAR)
        # Formula: -2.0 × ((holds-5)/5)²
        #   5 holds → 0.00
        #   10 holds → -2.00
        #   15 holds → -8.00
        #   20 holds → -18.00
        passivity_penalty = 0.0

        if self.consecutive_holds > 5:
            # FIX 3: Quadratic penalty (exponential growth)
            excess_holds = self.consecutive_holds - 5
            passivity_penalty = -2.0 * ((excess_holds / 5.0) ** 2)
            passivity_penalty = max(passivity_penalty, -18.0)  # Cap at -18.0

            if self.verbose and self.consecutive_holds % 5 == 0:
                self.log(f"   [HOLD PENALTY V2.7] {self.consecutive_holds} consecutive holds → Penalty {passivity_penalty:.2f}")

        reward += passivity_penalty

        # =====================================================================
        # ACTION REWARD (NEW!) - Reward for taking action (not HOLD)
        # =====================================================================
        # Small bonus just for BUY/SELL action (encourages exploration)
        action_reward = 0.0
        if len(self.action_history) > 0:
            last_discrete_action = self.action_history[-1]
            if last_discrete_action != 1:  # Not HOLD
                action_reward = +0.05  # Small reward for taking action

        # BONUS: Reward for changing action (fights mode collapse!)
        if len(self.action_history) >= 2:
            current_action = self.action_history[-1]
            previous_action = self.action_history[-2]
            if current_action != previous_action:
                action_reward += 0.03  # Bonus for changing direction/action

        reward += action_reward

        # =====================================================================
        # F4: CRITIC VARIANCE BONUS (Reward learning to distinguish states)
        # =====================================================================
        # We can't directly access Critic outputs in env, so we use REWARD VARIANCE
        # as a proxy: High reward variance = Agent experiencing diverse outcomes
        # This indirectly encourages Critic to learn meaningful value differences
        # Standard: Citadel, Two Sigma (value function learning incentives)
        critic_proxy_bonus = 0.0

        # Track reward history for variance calculation
        self.value_history.append(reward)  # Use value_history deque for rewards

        if len(self.value_history) >= 100:
            reward_std = np.std(list(self.value_history))

            # If reward variance is HIGH (>1.0) → Agent is experiencing diverse situations
            # This means Critic CAN learn to distinguish states → Bonus!
            if reward_std > 1.0:
                critic_proxy_bonus = +0.5
                if self.verbose and self.total_actions % 500 == 0:
                    self.log(f"   [CRITIC VARIANCE] Reward std: {reward_std:.2f} (>1.0) → Bonus +0.5")

        reward += critic_proxy_bonus

        # =====================================================================
        # APPLY REWARD SCALE (progressive based on trade count)
        # =====================================================================
        reward *= reward_scale  # 0.3, 0.6, or 1.0 based on trade count

        # =====================================================================
        # FIX 1 V2.7 NUCLEAR: Add TRADING ACTION REWARDS at END (PROTECTED!)
        # =====================================================================
        # Add trading_action_reward AFTER scaling to ensure it's NOT diluted
        # This is CRITICAL: +5.0 must be preserved to encourage trading
        reward += trading_action_reward

        return float(reward)

    def _calculate_diversity_score(self) -> float:
        """Calculate action diversity (Shannon entropy)"""
        # FIX: Reduced from 10 to 5 actions (faster feedback)
        if len(self.recent_actions) < 5:
            return 0.0

        action_counts = Counter(self.recent_actions)
        total = len(self.recent_actions)

        entropy = 0.0
        for count in action_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)

        # Normalize (max entropy for 3 actions = log(3) = 1.0986)
        max_entropy = np.log(3.0)
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        # Target: > 0.7
        if diversity > 0.7:
            return 0.5
        elif diversity > 0.5:
            return 0.2
        else:
            return -0.2  # Penalty for low diversity

    def _calculate_regret(self) -> float:
        """
        Calculate regret signal - missed opportunities (Agent 8 Mean Reversion)

        For Mean Reversion:
        - Regret if we didn't enter at optimal reversal point
        - Regret if we exited too early before full retracement

        Returns:
            float: Regret signal (-1 to +1)
                +1 = high regret (missed big opportunity)
                0 = no regret
                -1 = negative regret (avoided bad trade)
        """
        if len(self.trades) < 5:
            return 0.0

        # Analyze last 5 trades
        recent_trades = self.trades[-5:]
        total_regret = 0.0

        for trade in recent_trades:
            entry_idx = self.features_df.index.get_loc(
                self.features_df.index[min(trade.get('entry_step', self.current_step), self.current_step)]
            )
            exit_idx = self.features_df.index.get_loc(
                self.features_df.index[min(trade.get('exit_step', self.current_step), self.current_step)]
            )

            # Get price action during trade
            trade_prices = self.prices_df.iloc[entry_idx:exit_idx+1]

            if len(trade_prices) < 2:
                continue

            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            actual_pnl = trade['pnl']

            # Calculate optimal PnL (if perfect timing)
            if trade['position_side'] == 1:  # LONG
                optimal_entry = trade_prices['low'].min()  # Best entry (lowest)
                optimal_exit = trade_prices['high'].max()  # Best exit (highest)
            else:  # SHORT
                optimal_entry = trade_prices['high'].max()  # Best entry (highest)
                optimal_exit = trade_prices['low'].min()  # Best exit (lowest)

            # Calculate optimal PnL
            optimal_pnl = abs(optimal_exit - optimal_entry) * self.XAUUSD_CONTRACT_SIZE * trade['position_size']

            # Regret = (optimal - actual) / optimal
            if optimal_pnl > 0:
                regret = (optimal_pnl - actual_pnl) / optimal_pnl
                total_regret += regret

        # Average regret
        avg_regret = total_regret / len(recent_trades)

        # Clip to [-1, 1]
        return np.clip(avg_regret, -1.0, 1.0)

    def _classify_market_regime(self) -> int:
        """
        Classify market regime for Mean Reversion strategy (Agent 8)

        Returns:
            0 = Ranging (IDEAL for Mean Reversion)
            1 = Trending (AVOID Mean Reversion)
            2 = Volatile (RISKY for Mean Reversion)
        """
        # Need ATR for volatility
        if 'atr_14_m15' not in self.features_df.columns and 'atr_14' not in self.features_df.columns:
            return 0  # Default to ranging

        # Get ATR column (try M15 specific first, then generic)
        atr_col = 'atr_14_m15' if 'atr_14_m15' in self.features_df.columns else 'atr_14'

        current_atr = self.features_df[atr_col].iloc[self.current_step]

        # Calculate ATR MA (last 20 bars)
        start_idx = max(0, self.current_step - 20)
        atr_ma = self.features_df[atr_col].iloc[start_idx:self.current_step+1].mean()

        # VOLATILE if ATR > 1.5x MA (high volatility)
        if current_atr > atr_ma * 1.5:
            return 2

        # Check for TRENDING vs RANGING
        # Use SMA50 vs price distance
        if 'sma_50_m15' in self.features_df.columns:
            sma_col = 'sma_50_m15'
        elif 'sma_50' in self.features_df.columns:
            sma_col = 'sma_50'
        else:
            # No SMA available, use ATR only
            return 0 if current_atr <= atr_ma else 1

        current_price = self.prices_df['close'].iloc[self.current_step]
        sma = self.features_df[sma_col].iloc[self.current_step]

        # TRENDING if price > 2% from SMA50
        price_distance = abs(current_price - sma) / sma if sma > 0 else 0
        if price_distance > 0.02:  # 2% threshold
            return 1  # Trending

        # RANGING (ideal for Mean Reversion)
        return 0

    def _hours_until_macro_event(self) -> float:
        """
        Calculate hours until next major macro event (Agent 8 Mean Reversion)

        Major events for Gold:
        - Friday 8:30 AM EST (13:30 UTC): NFP (Non-Farm Payrolls)
        - Wednesday ~14:00 EST (19:00 UTC): FOMC announcements (when scheduled)

        Mean Reversion avoids trading near these events.

        Returns:
            float: Hours until next major event (0 to 168)
        """
        current_datetime = self.features_df.index[self.current_step]
        day_of_week = current_datetime.weekday()  # 0=Monday, 4=Friday
        hour_utc = current_datetime.hour
        minute = current_datetime.minute

        # Convert to decimal hours
        current_hour_decimal = hour_utc + minute / 60.0

        # Friday NFP (13:30 UTC = 8:30 AM EST)
        if day_of_week == 4:  # Friday
            nfp_hour = 13.5  # 13:30 UTC
            if current_hour_decimal < nfp_hour:
                # NFP today
                return nfp_hour - current_hour_decimal
            else:
                # NFP next Friday (7 days)
                hours_until_next_friday = (7 * 24) - (current_hour_decimal - nfp_hour)
                return hours_until_next_friday
        else:
            # Calculate hours until next Friday
            days_until_friday = (4 - day_of_week) % 7
            if days_until_friday == 0:
                days_until_friday = 7  # Next week
            hours_until_friday = days_until_friday * 24 + (13.5 - current_hour_decimal)
            return max(0, hours_until_friday)

    def _volatility_percentile(self) -> float:
        """
        Calculate current volatility percentile (Agent 8 Mean Reversion)

        Mean Reversion works best in LOW to MEDIUM volatility.
        High volatility = risky for Mean Reversion.

        Returns:
            float: Volatility percentile (0 to 1)
                0 = lowest volatility (100 bars)
                1 = highest volatility (100 bars)
        """
        # Get ATR column
        if 'atr_14_m15' not in self.features_df.columns and 'atr_14' not in self.features_df.columns:
            return 0.5  # Default neutral

        atr_col = 'atr_14_m15' if 'atr_14_m15' in self.features_df.columns else 'atr_14'

        # Get last 100 bars of ATR
        start_idx = max(0, self.current_step - 100)
        end_idx = self.current_step + 1
        atr_history = self.features_df[atr_col].iloc[start_idx:end_idx].values

        if len(atr_history) < 10:
            return 0.5  # Not enough data

        current_atr = atr_history[-1]

        # Calculate percentile (how many values are BELOW current)
        percentile = (atr_history < current_atr).sum() / len(atr_history)

        return percentile

    def _update_adaptive_reward_multiplier(self):
        """
        PHASE 2: CURRICULUM LEARNING (Adaptive Reward Scaling)

        Ajuste le reward multiplier basé sur la performance récente:
        - Bonne performance (Sharpe > 1.0, Win Rate > 55%) → Multiplier UP (max 2.0x)
        - Mauvaise performance (Sharpe < 0.5, Win Rate < 45%) → Multiplier DOWN (min 0.5x)

        Standard: Renaissance Technologies (Medallion Fund - adaptive learning)
        Paper: "Curriculum Learning for RL" (Narvekar et al., 2020)
        """
        # Need at least 10 trades and 30 days of data
        if len(self.trades) < 10 or len(self.daily_returns) < 30:
            self.adaptive_reward_multiplier = 1.0  # Neutral during warmup
            return

        # Calculate performance metrics
        performance_score = 0.0

        # 1. Sharpe Ratio (50% weight)
        returns_arr = np.array(list(self.daily_returns))
        if returns_arr.std() > 0:
            sharpe = (returns_arr.mean() / returns_arr.std()) * np.sqrt(252)
            # Sharpe > 1.5 = excellent (+0.5)
            # Sharpe > 1.0 = good (+0.25)
            # Sharpe > 0.5 = okay (0)
            # Sharpe < 0.5 = poor (-0.25)
            if sharpe > 1.5:
                performance_score += 0.5
            elif sharpe > 1.0:
                performance_score += 0.25
            elif sharpe < 0.5:
                performance_score -= 0.25

        # 2. Win Rate (30% weight)
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = winning_trades / len(self.trades)
        # Win rate > 60% = excellent (+0.3)
        # Win rate > 55% = good (+0.15)
        # Win rate > 50% = okay (0)
        # Win rate < 45% = poor (-0.15)
        if win_rate > 0.60:
            performance_score += 0.3
        elif win_rate > 0.55:
            performance_score += 0.15
        elif win_rate < 0.45:
            performance_score -= 0.15

        # 3. Drawdown Control (20% weight)
        # Low DD = good, high DD = bad
        if self.max_drawdown < 0.05:  # < 5% = excellent
            performance_score += 0.2
        elif self.max_drawdown < 0.08:  # < 8% = good
            performance_score += 0.1
        elif self.max_drawdown > 0.10:  # > 10% = bad
            performance_score -= 0.2

        # Convert performance_score to multiplier (0.5x to 2.0x)
        # performance_score range: -0.6 to +1.0
        # Map to: 0.5x to 2.0x
        self.adaptive_reward_multiplier = np.clip(1.0 + performance_score, 0.5, 2.0)

        # Store for tracking
        self.performance_score = performance_score

    def _calculate_trade_similarity_score(self) -> float:
        """
        WALL STREET V2: Trade Quality Memory System (25+25 INSTITUTIONAL GRADE)

        Compare current market context with:
        - 25 best trades (WINNERS - highest PnL)
        - 25 worst trades (LOSERS - lowest PnL)

        Uses COSINE SIMILARITY (like Agent 7) for fast pattern matching.

        Returns:
            float: Trade Similarity Score [-1.0, +1.0]
                +1.0 = Context highly similar to WINNERS (GO AHEAD!)
                0.0  = Neutral (standard confidence)
                -1.0 = Context highly similar to LOSERS (AVOID!)

        Used by: Renaissance Technologies, Two Sigma (pattern recognition)
        Standard: Like Agent 7, but adapted for Mean Reversion M15
        """
        # =====================================================================
        # PHASE 1: CHECK IF MEMORY IS ACTIVE (After 10 trades)
        # =====================================================================
        if not self.memory_active:
            return 0.0  # Memory not active yet (need 10 trades minimum)

        # Minimum patterns required for meaningful comparison
        min_patterns = max(5, self.memory_capacity // 2)  # At least half capacity
        if len(self.winner_patterns) < min_patterns or len(self.loser_patterns) < min_patterns:
            return 0.0  # Not enough patterns yet

        # =====================================================================
        # PHASE 2: EXTRACT CURRENT 15 FEATURES
        # =====================================================================
        try:
            current_features = self.entry_features if self.entry_features is not None else np.zeros(15, dtype=np.float32)
        except:
            return 0.0  # Fallback to neutral if features unavailable

        # Safety check: ensure features are valid
        if np.all(current_features == 0) or np.any(np.isnan(current_features)):
            return 0.0

        # =====================================================================
        # PHASE 3: EXTRACT WINNER FEATURES FROM MEMORY
        # =====================================================================
        # Use stored winner_patterns (already sorted, top N best)
        winner_features_list = []
        for pattern in self.winner_patterns:
            if 'entry_features' in pattern and pattern['entry_features'] is not None:
                try:
                    features = np.array(pattern['entry_features'], dtype=np.float32)
                    if len(features) == 15 and not np.any(np.isnan(features)):
                        winner_features_list.append(features)
                except:
                    continue

        if len(winner_features_list) == 0:
            return 0.0

        winner_features = np.array(winner_features_list, dtype=np.float32)

        # =====================================================================
        # PHASE 4: EXTRACT LOSER FEATURES FROM MEMORY
        # =====================================================================
        # Use stored loser_patterns (already sorted, bottom N worst)
        loser_features_list = []
        for pattern in self.loser_patterns:
            if 'entry_features' in pattern and pattern['entry_features'] is not None:
                try:
                    features = np.array(pattern['entry_features'], dtype=np.float32)
                    if len(features) == 15 and not np.any(np.isnan(features)):
                        loser_features_list.append(features)
                except:
                    continue

        if len(loser_features_list) == 0:
            return 0.0

        loser_features = np.array(loser_features_list, dtype=np.float32)

        # =====================================================================
        # PHASE 5: COSINE SIMILARITY CALCULATION (Agent 7 Method)
        # =====================================================================
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            """
            Fast cosine similarity for 15-dimensional vectors.

            Formula: cos(θ) = (a · b) / (||a|| × ||b||)
            Range: [-1, 1]
                1 = identical direction
                0 = orthogonal
               -1 = opposite direction
            """
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(np.dot(a, b) / (norm_a * norm_b))

        # =====================================================================
        # PHASE 6: AVERAGE SIMILARITY WITH WINNERS
        # =====================================================================
        similarities_winners = []
        for wf in winner_features:
            sim = cosine_sim(current_features, wf)
            if not np.isnan(sim):
                similarities_winners.append(sim)

        sim_winners = np.mean(similarities_winners) if len(similarities_winners) > 0 else 0.0

        # =====================================================================
        # PHASE 7: AVERAGE SIMILARITY WITH LOSERS
        # =====================================================================
        similarities_losers = []
        for lf in loser_features:
            sim = cosine_sim(current_features, lf)
            if not np.isnan(sim):
                similarities_losers.append(sim)

        sim_losers = np.mean(similarities_losers) if len(similarities_losers) > 0 else 0.0

        # =====================================================================
        # PHASE 8: TRADE SIMILARITY SCORE = Winners - Losers
        # =====================================================================
        # Interpretation:
        #   +1.0 = current context looks like winners, NOT like losers → HIGH confidence
        #   0.0  = neutral, can't tell
        #   -1.0 = current context looks like losers, NOT like winners → LOW confidence
        trade_similarity = sim_winners - sim_losers

        # Clip to valid range
        return float(np.clip(trade_similarity, -1.0, 1.0))

    def _update_memory_capacity(self):
        """
        Update memory capacity progressively based on trade count

        Growth schedule:
        - 10-19 trades: capacity = 10 (activation!)
        - 20-29 trades: capacity = 15
        - 30-39 trades: capacity = 20
        - 40+ trades:   capacity = 25 (max)
        """
        total_trades = len(self.trades)

        if total_trades >= 40:
            self.memory_capacity = 25
        elif total_trades >= 30:
            self.memory_capacity = 20
        elif total_trades >= 20:
            self.memory_capacity = 15
        elif total_trades >= 10:
            self.memory_capacity = 10
            self.memory_active = True  # Activate memory!
        else:
            self.memory_active = False
            self.memory_capacity = 0

    def _store_trade_pattern(self, trade_dict: dict):
        """
        Store trade pattern in memory (PROGRESSIVE GROWTH 10→15→20→25)

        Uses 15 entry features stored in self.entry_features:
        - Technical indicators (RSI, MACD, ADX, Stochastic, etc.)
        - Volatility (ATR, BB width)
        - Market context (regime, volatility percentile, drawdown)

        Stores in:
        - winner_patterns: TOP N best trades (sorted by PnL%)
        - loser_patterns: BOTTOM N worst trades (sorted by PnL%)

        N = memory_capacity (grows 10→15→20→25)
        """
        # Update memory capacity first
        self._update_memory_capacity()

        # Don't store if memory not active yet
        if not self.memory_active:
            return

        # Store pattern with entry features (15 features)
        pattern = {
            'entry_features': self.entry_features.copy() if self.entry_features is not None else np.zeros(15),
            'pnl': trade_dict['pnl'],
            'pnl_pct': trade_dict['pnl_pct'],
            'position_side': trade_dict['position_side'],
            'hold_duration': trade_dict['hold_duration'],
        }

        # Add to appropriate list
        if trade_dict['pnl_pct'] > 0:  # WINNER
            self.winner_patterns.append(pattern)
            # Sort by PnL% (descending) and keep top N
            self.winner_patterns = sorted(self.winner_patterns, key=lambda x: x['pnl_pct'], reverse=True)[:self.memory_capacity]
        else:  # LOSER
            self.loser_patterns.append(pattern)
            # Sort by PnL% (ascending) and keep bottom N (worst)
            self.loser_patterns = sorted(self.loser_patterns, key=lambda x: x['pnl_pct'])[:self.memory_capacity]

    def _calculate_win_probability(self) -> float:
        """
        Calculate probability that CURRENT setup will result in winning trade

        Uses pattern matching against historical winners/losers:
        - High similarity to winners → High probability
        - High similarity to losers → Low probability

        Returns:
            float: Win probability (0 to 1)
                1.0 = Very similar to past winners
                0.0 = Very similar to past losers
                0.5 = Neutral (no patterns yet)
        """
        # Need at least 10 patterns total
        total_patterns = len(self.winner_patterns) + len(self.loser_patterns)
        if total_patterns < 10:
            return 0.5  # Default neutral

        # Extract current features
        current_pattern = {
            'market_regime': self.market_regime,
        }

        if 'rsi_14_m15' in self.features_df.columns or 'rsi_14' in self.features_df.columns:
            rsi_col = 'rsi_14_m15' if 'rsi_14_m15' in self.features_df.columns else 'rsi_14'
            current_pattern['rsi'] = float(self.features_df[rsi_col].iloc[self.current_step])

        if 'atr_14_m15' in self.features_df.columns or 'atr_14' in self.features_df.columns:
            atr_col = 'atr_14_m15' if 'atr_14_m15' in self.features_df.columns else 'atr_14'
            current_pattern['atr'] = float(self.features_df[atr_col].iloc[self.current_step])

        # Calculate similarity to winners
        winner_similarities = []
        for winner in self.winner_patterns[-50:]:  # Last 50 winners
            similarity = self._pattern_similarity(current_pattern, winner)
            winner_similarities.append(similarity)

        # Calculate similarity to losers
        loser_similarities = []
        for loser in self.loser_patterns[-50:]:  # Last 50 losers
            similarity = self._pattern_similarity(current_pattern, loser)
            loser_similarities.append(similarity)

        # Average similarities
        avg_winner_sim = np.mean(winner_similarities) if winner_similarities else 0.0
        avg_loser_sim = np.mean(loser_similarities) if loser_similarities else 0.0

        # Win probability = winner_sim / (winner_sim + loser_sim)
        total_sim = avg_winner_sim + avg_loser_sim
        if total_sim > 0:
            win_prob = avg_winner_sim / total_sim
        else:
            win_prob = 0.5

        # Clip to [0.2, 0.8] (never 0% or 100%)
        return np.clip(win_prob, 0.2, 0.8)

    def _pattern_similarity(self, pattern1: dict, pattern2: dict) -> float:
        """
        Calculate similarity between two trade patterns

        Returns:
            float: Similarity score (0 to 1)
                1.0 = identical patterns
                0.0 = completely different
        """
        similarity = 0.0
        features_compared = 0

        # Market regime (exact match)
        if 'market_regime' in pattern1 and 'market_regime' in pattern2:
            if pattern1['market_regime'] == pattern2['market_regime']:
                similarity += 0.4  # 40% weight
            features_compared += 1

        # RSI (numerical similarity)
        if 'rsi' in pattern1 and 'rsi' in pattern2:
            rsi_diff = abs(pattern1['rsi'] - pattern2['rsi'])
            rsi_sim = max(0, 1.0 - (rsi_diff / 100.0))  # 0-100 scale
            similarity += rsi_sim * 0.4  # 40% weight
            features_compared += 1

        # ATR (normalized similarity)
        if 'atr' in pattern1 and 'atr' in pattern2:
            atr1 = pattern1['atr']
            atr2 = pattern2['atr']
            if atr1 > 0 and atr2 > 0:
                atr_ratio = min(atr1, atr2) / max(atr1, atr2)
                similarity += atr_ratio * 0.2  # 20% weight
                features_compared += 1

        # Normalize by features compared
        if features_compared > 0:
            return similarity / features_compared
        else:
            return 0.5

    def _calculate_kelly_fraction(self) -> float:
        """
        Calculate Kelly Criterion optimal position size fraction.

        Formula: f* = (p × b - q) / b
        Where:
        - p = win rate (probability of win)
        - q = 1 - p (probability of loss)
        - b = win/loss ratio (avg_win / avg_loss)
        - f* = optimal fraction of capital to risk

        Used by: Bridgewater, Renaissance Technologies
        Standard: Fractional Kelly (0.25x to 0.5x) for safety

        Returns:
            float: Kelly fraction (0.0 to 1.0)
                0.25 = Fractional Kelly (conservative)
                0.5 = Half Kelly (balanced)
                1.0 = Full Kelly (aggressive, NOT recommended)
        """
        # =====================================================================
        # PHASE 1: MINIMUM TRADES REQUIRED (Need statistics)
        # =====================================================================
        if len(self.trades) < 20:
            return 0.5  # Default neutral (50% of max risk) during warmup

        # =====================================================================
        # PHASE 2: CALCULATE WIN RATE (p)
        # =====================================================================
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]

        if len(winning_trades) == 0 or len(losing_trades) == 0:
            return 0.5  # Need both wins and losses for Kelly

        p = len(winning_trades) / len(self.trades)  # Win rate
        q = 1 - p  # Loss rate

        # =====================================================================
        # PHASE 3: CALCULATE WIN/LOSS RATIO (b)
        # =====================================================================
        avg_win = np.mean([abs(t['pnl']) for t in winning_trades])
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades])

        if avg_loss == 0:
            return 0.5  # Avoid division by zero

        b = avg_win / avg_loss  # Win/loss ratio

        # =====================================================================
        # PHASE 4: KELLY FORMULA
        # =====================================================================
        # f* = (p × b - q) / b
        kelly_fraction = (p * b - q) / b

        # =====================================================================
        # PHASE 5: FRACTIONAL KELLY (0.25x - INSTITUTIONAL STANDARD)
        # =====================================================================
        # Full Kelly is too aggressive and leads to ruin
        # Fractional Kelly (0.25x or 0.5x) reduces variance
        #
        # Standard:
        # - Renaissance Technologies: 0.25x Kelly (very conservative)
        # - Bridgewater: 0.33x Kelly
        # - Two Sigma: 0.5x Kelly
        #
        # We use 0.25x for FTMO compliance (max 10% DD)
        fractional_kelly = kelly_fraction * 0.25

        # =====================================================================
        # PHASE 6: CLIP TO VALID RANGE
        # =====================================================================
        # Kelly can be negative if edge is negative (p × b < q)
        # In that case, Kelly suggests NO position (bet 0%)
        # We clip to [0.2, 1.0] to allow exploration even with negative Kelly
        fractional_kelly = np.clip(fractional_kelly, 0.2, 1.0)

        return float(fractional_kelly)

    def _calculate_var_95(self) -> float:
        """
        Calculate VaR 95% (Value at Risk - 95th percentile).

        VaR 95% = 5th percentile of daily returns
        Interpretation: "In 95% of cases, daily loss won't exceed VaR"

        Standard:
        - J.P. Morgan RiskMetrics
        - All institutional traders (Citadel, Two Sigma, etc.)

        Returns:
            float: VaR 95% (negative value)
                -1.5% = Good (95% of days, loss < 1.5%)
                -2.5% = Warning (high volatility)
                -3.0%+ = RED FLAG (excessive risk)
        """
        # =====================================================================
        # PHASE 1: MINIMUM 30 DAILY RETURNS REQUIRED
        # =====================================================================
        if len(self.daily_returns) < 30:
            return 0.0  # Not enough data for reliable VaR

        # =====================================================================
        # PHASE 2: CONVERT TO NUMPY ARRAY
        # =====================================================================
        returns_arr = np.array(list(self.daily_returns))

        # =====================================================================
        # PHASE 3: CALCULATE 5TH PERCENTILE (VaR 95%)
        # =====================================================================
        # np.percentile(data, 5) = 5th percentile
        # This is the threshold below which 5% of returns fall
        var_95 = np.percentile(returns_arr, 5)

        # =====================================================================
        # PHASE 4: RETURN (Should be negative if losses occurred)
        # =====================================================================
        return float(var_95)

    def _calculate_tail_risk(self) -> bool:
        """
        Detect tail risk using Kurtosis (fat tails detection).

        Kurtosis measures "tailedness" of distribution:
        - Kurtosis = 3.0 → Normal distribution (thin tails)
        - Kurtosis > 3.0 → Leptokurtic (fat tails - DANGER!)
        - Kurtosis < 3.0 → Platykurtic (thin tails)

        Fat tails = Black Swan events more likely than normal distribution predicts

        Standard:
        - Taleb "The Black Swan" (2007)
        - Institutional risk management

        Returns:
            bool: True if tail risk detected (kurtosis > 3.0)
        """
        # =====================================================================
        # PHASE 1: MINIMUM 30 DAILY RETURNS REQUIRED
        # =====================================================================
        if len(self.daily_returns) < 30:
            return False  # Not enough data

        # =====================================================================
        # PHASE 2: CONVERT TO NUMPY ARRAY
        # =====================================================================
        returns_arr = np.array(list(self.daily_returns))

        # =====================================================================
        # PHASE 3: CALCULATE KURTOSIS (Excess Kurtosis)
        # =====================================================================
        # scipy.stats.kurtosis calculates EXCESS kurtosis (kurtosis - 3)
        # Normal distribution has excess kurtosis = 0
        # So: excess_kurtosis > 0 → fat tails
        from scipy.stats import kurtosis
        try:
            excess_kurt = kurtosis(returns_arr, fisher=True)  # Fisher=True returns excess kurtosis
        except:
            return False  # Fallback if calculation fails

        # =====================================================================
        # PHASE 4: TAIL RISK THRESHOLD
        # =====================================================================
        # Excess kurtosis > 0 → fat tails (kurtosis > 3.0)
        # We use threshold of 1.0 to be conservative
        # (i.e., kurtosis > 4.0 to trigger tail risk warning)
        tail_risk = excess_kurt > 1.0

        return tail_risk

    def _calculate_adaptive_base_risk(self, confidence: float, atr: float) -> float:
        """
        Calculate adaptive base risk (0.33% to 1.0%) with Kelly Criterion.

        V2 ULTIMATE: Integrates Kelly optimal sizing with confidence-based adjustment.

        Returns:
            float: Base risk percentage (0.0033 to 0.01)
        """
        # =====================================================================
        # PHASE 1: CONFIDENCE FACTOR (0 to 1)
        # =====================================================================
        conf_min = self.MIN_CONFIDENCE_THRESHOLD
        conf_max = 1.0
        conf_factor = (confidence - conf_min) / (conf_max - conf_min)
        conf_factor = max(0.0, min(1.0, conf_factor))

        # =====================================================================
        # PHASE 2: KELLY FRACTION (0.2 to 1.0)
        # =====================================================================
        kelly_fraction = self._calculate_kelly_fraction()

        # =====================================================================
        # PHASE 3: COMBINE CONFIDENCE + KELLY
        # =====================================================================
        # Base risk range: 0.33% to 1.0%
        # Multiply by Kelly fraction to optimize sizing
        #
        # Example:
        # - Confidence 100%, Kelly 1.0 → base_risk = 1.0%
        # - Confidence 100%, Kelly 0.5 → base_risk = 0.665%
        # - Confidence 50%, Kelly 1.0 → base_risk = 0.665%
        # - Confidence 50%, Kelly 0.5 → base_risk = 0.4975%
        base_risk = 0.0033 + (conf_factor * 0.0067)  # 0.33% to 1.0%
        optimized_risk = base_risk * kelly_fraction

        # =====================================================================
        # PHASE 4: CLIP TO FTMO LIMITS
        # =====================================================================
        # FTMO max risk per trade: 1%
        # We clip to [0.001, 0.01] (0.1% to 1.0%)
        optimized_risk = np.clip(optimized_risk, 0.001, 0.01)

        return float(optimized_risk)

    def _calculate_risk_multiplier(self) -> float:
        """Calculate risk multiplier based on DD (production only)"""
        if self.training_mode:
            return 1.0  # Always 1.0 in training

        # Production: reduce risk progressively from 7% to 10% DD
        dd = self.max_drawdown
        if dd < 0.07:
            return 1.0
        elif dd >= 0.10:
            return 0.0
        else:
            # Linear: 7% → 1.0, 10% → 0.0
            return max(0.0, (0.10 - dd) / 0.03)

    def _get_atr(self) -> float:
        """Get current ATR (simplified)"""
        if 'atr_14' in self.features_df.columns:
            atr = self.features_df['atr_14'].iloc[self.current_step]
            return max(1.0, atr)  # Min 1.0
        else:
            # Fallback: calculate from prices
            start_idx = max(0, self.current_step - 14)
            end_idx = self.current_step + 1
            recent_prices = self.prices_df.iloc[start_idx:end_idx]

            tr_values = []
            for i in range(1, len(recent_prices)):
                high = recent_prices['high'].iloc[i]
                low = recent_prices['low'].iloc[i]
                close_prev = recent_prices['close'].iloc[i-1]
                tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
                tr_values.append(tr)

            atr = np.mean(tr_values) if tr_values else 10.0
            return max(1.0, atr)

    def _get_observation(self) -> np.ndarray:
        """Get current observation (209 base + 13 RL = 222 features)"""
        # Base features (209 = 199 + 10 temporal V3)
        base_features = self.features_df.iloc[self.current_step].values.astype(np.float32)

        # RL features (13)
        rl_features = []

        # 1-5. Last 5 actions (one-hot or normalized)
        actions_list = list(self.action_history) + [1] * (5 - len(self.action_history))
        for a in actions_list[-5:]:
            rl_features.append(float(a) / 2.0)  # Normalize 0-2 to 0-1

        # 6. Regret signal
        rl_features.append(self.regret_signal)

        # 7. Position duration (normalized)
        duration = (self.current_step - self.position_entry_step) / 100.0 if self.position_side != 0 else 0.0
        rl_features.append(duration)

        # 8. Unrealized PnL ratio
        pnl_ratio = self.unrealized_pnl / self.initial_risk_amount if self.initial_risk_amount > 0 else 0.0
        rl_features.append(pnl_ratio)

        # 9. Market regime (0-2)
        rl_features.append(float(self.market_regime) / 2.0)

        # 10. Hours until event (normalized)
        rl_features.append(self.hours_until_event / 1000.0)

        # 11. Volatility percentile
        rl_features.append(self.volatility_percentile)

        # 12. Position side
        rl_features.append(float(self.position_side))

        # 13. Trade similarity score (pattern memory)
        rl_features.append(self.trade_similarity_score)

        # Combine
        rl_features_array = np.array(rl_features, dtype=np.float32)
        full_observation = np.concatenate([base_features, rl_features_array])

        # NaN GUARD: Replace NaN/Inf with 0 (critical for SAC stability)
        full_observation = np.nan_to_num(full_observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return full_observation

    def _get_info(self) -> Dict:
        """Get info dict"""
        current_date = self.features_df.index[self.current_step]
        current_price = self.prices_df['close'].iloc[self.current_step]

        return {
            'step': self.current_step,
            'date': current_date,
            'price': current_price,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'position_side': self.position_side,
            'unrealized_pnl': self.unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'total_trades': len(self.trades),
            'ftmo_violations': self.ftmo_violations,
            'risk_multiplier': self.risk_multiplier,
        }

    def render(self, mode='human'):
        """Render environment (optional)"""
        if mode == 'human':
            info = self._get_info()
            print(f"\nStep: {info['step']} | Date: {info['date']}")
            print(f"Balance: ${info['balance']:.2f} | Equity: ${info['equity']:.2f}")
            print(f"Position: {['SHORT', 'FLAT', 'LONG'][info['position_side']+1]}")
            print(f"Trades: {info['total_trades']} | DD: {info['max_drawdown']:.2%}")

    def close(self):
        """Close environment"""
        pass


# Factory function for easy environment creation
def make_agent8_env(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    initial_balance: float = 100000.0,
    training_mode: bool = True,
    verbose: bool = False
) -> GoldTradingEnvAgent8:
    """
    Factory function to create Agent 8 trading environment

    Args:
        features_df: DataFrame with 209 base features (199+10 temporal V3)
        prices_df: DataFrame with OHLCV prices (M15)
        initial_balance: Starting balance
        training_mode: True for training, False for backtest/live
        verbose: Print detailed logs

    Returns:
        GoldTradingEnvAgent8 instance
    """
    return GoldTradingEnvAgent8(
        features_df=features_df,
        prices_df=prices_df,
        initial_balance=initial_balance,
        training_mode=training_mode,
        verbose=verbose
    )

# Alias for backward compatibility with test scripts
make_env = make_agent8_env
