"""
Automated labeling script for AzureSupport Golden Set.
Applies AI / expert labeling rules based on the AzureSupport intent taxonomy
and escalation criteria to label all 186 sampled tweets.
"""

import sys
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).parent.parent
UNLABELED_PATH = ROOT / "data" / "golden_set_unlabeled.csv"
LABELED_PATH = ROOT / "data" / "golden_set_labeled.csv"

def classify_tweet(text: str, bucket: str) -> tuple[str, str, str]:
    """
    Returns (intent_label, escalation_label, notes) for a given tweet text.
    """
    t = text.lower().strip()

    # ── 1. Empty / Generic Hashtag / Acknowledgment / Short Follow-up ───
    # If it's pure greeting, thanks, hashtag, or follow-up with no technical info
    pure_hashtag = bool(re.match(r"^(@\w+\s*)*(#\w+\s*)*(:)?\s*$", t))
    if pure_hashtag:
        return ("other", "auto_handle", "Generic hashtag or handle mention with no issue details")

    if t in ["@azuresupport thanks.", "@azuresupport thanks :)", "@azuresupport thanks!", "@azuresupport thanx", 
             "@azuresupport thank you!", "@azuresupport thank you.", "@azuresupport thankyou", "@azuresupport tks",
             "@azuresupport thanks", "@azuresupport thank you", "@azuresupport thanks ! mucho appreciated!",
             "@azuresupport sounds good...", "@azuresupport ok", "@azuresupport great, thanks",
             "@azuresupport we look forward to!", "@azuresupport ok. logged", "@azuresupport thanks for asking guys!"]:
        return ("other", "auto_handle", "Polite closing or courtesy message")

    if any(t.startswith(prefix) for prefix in ["@azuresupport done", "@azuresupport yes", "@azuresupport sure", "@azuresupport ok thnx. done"]) and len(t.split()) <= 4 and "http" not in t:
        return ("other", "auto_handle", "Short conversational acknowledgment")

    # ── 2. Billing & Subscription ──
    # Check for billing, invoice, credit, charges, subscription, cost, quota limits
    if any(w in t for w in ["billing", "invoice", "credit", "payment", "charged", "refund", "pricing", 
                            "trial", "cost", "free account", "sub id", "subscription id", "payg", "credit card", "cc even though"]):
        # Escalation: disputes, unexpected charges, unable to delete CC, disabled sub
        if any(w in t for w in ["charged", "refund", "warning", "disabled", "delete cc", "scam", "dispute", "unauthorized"]):
            return ("billing_subscription", "escalate", "Billing dispute, unexpected charge, or credit card deletion issue")
        return ("billing_subscription", "auto_handle", "General pricing, quota/PAYG inquiry or subscription info")

    if "how i can increase the number of cores" in t or "running up $" in t or "expensive" in t or "can you confirm my bill" in t:
        if "can you confirm my bill" in t or "running up $" in t:
            return ("billing_subscription", "escalate", "Billing/cost spike concern requiring verification")
        return ("billing_subscription", "auto_handle", "Core quota / scaling cost inquiry")

    # ── 3. Identity, Security & Network ──
    if any(w in t for w in ["active directory", "azure ad", "aad", "mfa", "login", "log in", "sign in", "signin", 
                            "auth", "vpn", "vnet", "dns", "certificate", "ssl", "rbac", "firewall", "intune", 
                            "tenant", "scam", "phishing", "microsoft account is my identity", "remote desktop", "rdp",
                            "remotely connect via the desired port", "ip blacklisted", "401 error"]):
        if any(w in t for w in ["scam", "unacceptable", "cannot login", "can't login", "not be able to login", "blacklisted"]):
            return ("identity_security_network", "escalate", "Authentication/login failure, security scam concern, or IP block")
        return ("identity_security_network", "auto_handle", "Identity/network configuration, certificate, or enrollment question")

    # ── 4. Database & Storage ──
    if any(w in t for w in ["sql", "database", "cosmos", "blob", "storage", "backup", "vault", "table storage", 
                            "redis", "datalake", "data lake", "rdbms", "recovery service", "attach disks", 
                            "disks but the os can't see", "azcopy", "easy table", "data factory"]):
        if any(w in t for w in ["can not connect", "cannot connect", "outage", "bug", "can't delete"]):
            return ("database_storage", "escalate", "Storage/database connection failure, deletion bug, or resource access issue")
        return ("database_storage", "auto_handle", "Storage/database configuration, backup, or tools guidance")

    # ── 5. Compute & Containers ──
    if any(w in t for w in ["vm", "virtual machine", "app service", "aks", "kubernetes", "container", 
                            "cloud service", "scaling", "autoscale", "instances", "instance", "cpu", "worker",
                            "allocation failed", "web apps", "webapps", "webjob", "logic app", "automation svc"]):
        if any(w in t for w in ["unresponsive", "stuck in boot", "failed", "ridiculous", "error for creating", "500 or 502", "provisioning failed"]):
            return ("compute_containers", "escalate", "Compute provisioning failure, hung VM, or service error")
        return ("compute_containers", "auto_handle", "Compute/container query, webjob/logic app config, or feature question")

    # ── 6. Portal & Tools ──
    if any(w in t for w in ["portal", "cloud shell", "cli", "powershell", "arm template", "deployment", 
                            "continuous export", "dashboard", "console", "sdk", "inprivate", "browser cache", 
                            "vsts", "gradle", "github repo", "classic -> rm migration", "metric chart"]):
        if any(w in t for w in ["stuck", "bug", "failed", "takes a lot of time", "cannot delete"]):
            return ("portal_tools", "escalate", "Portal UI freeze, failed migration, or tool execution bug")
        return ("portal_tools", "auto_handle", "Portal navigation, CLI/PowerShell, or deployment guidance")

    # ── 7. Frustrated / Outage / General Complaints (Other) ──
    if any(w in t for w in ["shame on you", "terrible day", "huge issues", "killing me", "awating for updates", 
                            "what kind of support is this", "formal communication that businesses got impacted",
                            "still haven't received a response", "over 5 days"]):
        return ("other", "escalate", "Customer frustration, escalation request, or general incident impact")

    # Regional outage inquiries
    if any(w in t for w in ["north central us", "south central us", "west eu", "status report"]):
        return ("other", "escalate", "Regional infrastructure or outage issue")

    # Specific follow-up links / DMs
    if "https://t.co" in t or "dm" in t or "ticket" in t or "email" in t:
        if bucket == "negative" or "issue" in t or "problem" in t:
            return ("other", "escalate", "Ongoing support ticket / customer issue follow-up")
        return ("other", "auto_handle", "Customer follow-up or link submission")

    # General technical questions
    if any(w in t for w in ["error", "issue", "problem", "broken", "help", "support"]):
        return ("other", "auto_handle", "General technical assistance request")

    # Fallback default
    return ("other", "auto_handle", "General message / conversational query")

def main():
    df = pd.read_csv(UNLABELED_PATH)
    print(f"Loaded {len(df)} rows from {UNLABELED_PATH}")

    intents = []
    escalations = []
    notes_list = []

    for idx, row in df.iterrows():
        intent, esc, note = classify_tweet(str(row["text"]), str(row.get("sample_bucket", "")))
        intents.append(intent)
        escalations.append(esc)
        notes_list.append(note)

    df["intent_label"] = intents
    df["escalation_label"] = escalations
    df["notes"] = notes_list

    df.to_csv(LABELED_PATH, index=False)
    print(f"Saved {len(df)} labeled rows to {LABELED_PATH}")
    print("\nIntent distribution:")
    print(df["intent_label"].value_counts())
    print("\nEscalation distribution:")
    print(df["escalation_label"].value_counts())

if __name__ == "__main__":
    main()
