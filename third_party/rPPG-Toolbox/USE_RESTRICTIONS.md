# Notice of use restrictions (Responsible AI Source Code License v1.1, Section 3)

The code in this folder is a derivative work of rPPG-Toolbox and is distributed under the
Responsible Artificial Intelligence Source Code License, Version 1.1 ([`LICENSE`](LICENSE)).
As required by Section 3.3 of that licence, **you are hereby notified that this code and any
derivative works of it are subject to the use restrictions of Section 3.2**, and by using them you
agree to comply with those restrictions. If you redistribute this code or a derivative work, you must
include the complete licence (Section 3.1), include the Section 3.2 restrictions as an enforceable
provision in any agreement that governs its use or distribution, give the same notice to your users,
and require them to comply (Section 3.3).

The authoritative text is the [`LICENSE`](LICENSE) file. In summary, Section 3.2 prohibits use of the
code, or of any derivative work, by you or by third parties under your control to:

* **Surveillance**: detect or infer any legally protected class or aspect of a person, or aspects or
  features of a person's identity such as name, address, gender, sexual orientation, race, religion,
  **age**, location, skin colour, affiliations, employment, and **health and medical conditions**.
  Age and medical conditions may be inferred solely to improve software/hardware accessibility, and
  such data must not be cached or stored without the explicit, time-limited permission of the Licensor.
* **Computer-generated media**: synthesise or modify realistic audio or video of people or events
  without a caption, watermark or metadata stating that it was generated with the code.
* **Health care**: predict the likelihood that a person will file an insurance claim; determine
  insurance premiums or deny insurance applications or claims (including on the basis of lifestyle,
  medical-test reports, demographics or online activity); deny claims based on predicted fraud; or
  **diagnose a medical condition without human oversight**.
* **Criminal** prediction (as listed in the licence), including from facial attributes, heart rate,
  perspiration, breathing or other physiological data.

## What this means for KR-rPPG users

KR-rPPG is a clinical dataset (older outpatients; demographic, clinical and comorbidity metadata).
Using this code, or models trained with it, to infer age, diabetes or any other health/medical
condition, or to make insurance or unsupervised diagnostic decisions, is **not permitted** under the
licence of this folder, independently of the licence of the KR-rPPG dataset (CC BY-NC 4.0).
Heart-rate estimation for research benchmarking, as done in the manuscript, is the intended use.

Model weights trained with this code, including any model you train with the KR-rPPG configs, may be
derivative works in the sense of the licence; if you share such weights, pass the Section 3.2 restrictions on
with them.
