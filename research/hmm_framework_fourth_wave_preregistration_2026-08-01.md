# HMM framework fourth-wave exploratory plan

Date frozen: 2026-08-01, before fourth-wave results.

The development, holdout, and proxy data have already been inspected in prior
waves. These tests are therefore mechanism exploration, not fresh confirmation.
Even a full pass cannot qualify production without a forward shadow.

Shared account gates remain unchanged from the third wave.

## Direction I: volatility-adaptive refit cadence

Keep the selected-order Gaussian model and all features. Refit every 63 anchored
business days when both QQQ and SMH lagged 21-day volatility are no greater than
their lagged 63-day volatility; refit every 21 anchored business days otherwise.
Use only prices strictly before the target date. The existing fast R38 layer remains
unchanged. This directly targets the observed stable-period benefit and early-crisis
failure of fixed 63-day refitting.

## Direction J: remove only SMH from macro state inputs

Remove SMH from HMM regime inputs while retaining QQQ, SPX, IEF, GLD, DBC, and UUP.
SMH remains an investable outcome and remains governed by all R38/R39 controls. This
is the smallest explicit separation between macro/growth state and semiconductor
idiosyncratic risk; no extra semiconductor model is introduced.
