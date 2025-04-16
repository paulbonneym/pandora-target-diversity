# pandora-target-diversity
Tools to ensure the target list is diverse. The main usage of this package is to narrow down a list of potential targets into a short list based on observability (transmission spectroscopy metric) and diversity in the parameter space. 

### Example Usage

Below is an example of generating a short list using 10000 iterations:

```python
from pandora-target-diversity import short_list
sl, div, tsm = short_list(10000)
```
The short list is returned as a pandas.DataFrame object as well as the diversity score and tsm score for the selected short list.

If you want to create a short list of a different length (20 by default), use the t argument:

```python
from pandora-target-diversity import short_list
sl_longer, div_longer, tsm_longer = short_list(10000,t=30)
```

The diversity score can also be tuned to change the acceptable level of diversity, which is 2 standard deviations by default:

```python
from pandora-target-diversity import short_list
#Create a list with less strict acceptable diversity
sl_less, div_less, tsm_less = short_list(10000,tun=1.5)
#Create a list with more strict acceptable diversity
sl_more, div_more, tsm_more = short_list(10000,tun=2.5)
```

If you would like to choose targets based on systems instead of individual planets, use the systems keyword:

```python
from pandora-target-diversity import short_list
sl_systems, div_systems, tsm_systems = short_list(10000,systems=True)
```
Note: in benchmark testing, this was around 9 times slower than the default. 

