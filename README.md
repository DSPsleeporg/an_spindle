# Averaged Neuron (AN) model

## Overview
AN model is a conductance-based (Hodgkin-Huxley type) neuron model using the mean-field approximation of groups of neurons.
The model can recapitulate the slow-wave sleep (SWS) firing pattern and the sleep spindle (SS) firing pattern. 

## Dependencies
The following packages need to be installed to use AN model:
- Python >= 3.5
- Numpy >= 1.10
- Scipy >= 1.0.0

## Usage (plotting the SWS firing pattern)
``` python
import matplotlib.pyplot as plt
import anmodel
an = anmodel.models.ANmodel()
an.set_sws_params()
s, _ = an.run_odeint()
plt.plot(s[4999:, 0])
```
![SWS firing (example)](images/sws_firing.png)

## References
### SWS (slow-wave sleep) firing pattern
- [Tatsuki et al., Neuron, 2016](https://www.cell.com/neuron/fulltext/S0896-6273(16)00169-0?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627316001690%3Fshowall%3Dtrue)
- [Rasmussen et al., Cell Systems, 2017](https://www.sciencedirect.com/science/article/pii/S2405471217305392)
- [Yoshida et al., PNAS, 2018](https://www.pnas.org/content/115/40/E9459.short)
### SS (sleep spindle) firing pattern
- [Yamada et al., iScience, 2021]()

## License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.