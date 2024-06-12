# FTools

Common tools for daily development.

## Installation

```bash
git clone https://github,com/nkfyz/Ftools.git
cd Ftools
python setup.py install
```

## Usage

Each tool can be loaded from Ftools directly.

### Ftimer

```python
import time
from Ftools import Ftimer

with Ftimer():
    time.sleep(5)
```
