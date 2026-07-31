"""Cloud training: rent a GPU, run the chain, get told, pay cents.

The desktop measurement that justifies this package: the GRPO reward is network-bound, so the
GPU idles 50-70% of every step — meaning the right rental is the *cheapest* card that fits the
policy (a $0.34/hr 4090-class), not the fastest. Workers stay on the flat-rate provider; the
rented box holds only the policy and trainer.

Design rules, non-negotiable:

* **Auto-terminate is structural.** Every path out of a run — success, failure, timeout, even the
  launcher crashing — ends at terminate. A rented GPU that outlives its job is a billing leak.
* **The budget ledger is a hard gate.** Provisioning past the monthly ceiling refuses with a
  complete reason (spent, ceiling, when it resets), exactly like the concurrency governor.
* **Public-ready.** No account ids, no keys, no machine paths. Secrets ride env vars documented
  in ``.env.example``; notifications are pluggable env-configured Telegram, not anyone's private
  daemon plumbing.
"""

from .budget import BudgetExceeded, BudgetLedger
from .providers import CloudProvider, GpuOffer, Instance, InstanceState, ProviderError

__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "CloudProvider",
    "GpuOffer",
    "Instance",
    "InstanceState",
    "ProviderError",
]
