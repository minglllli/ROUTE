from .cifar10 import get_cifar10
from .cifar100 import get_cifar100
from .eurosat import get_eurosat

from .oxfordpets import get_oxfordpets
DATASET_GETTERS = {'cifar10_set1': get_cifar10,
				   'cifar100_set1': get_cifar100,
				   'eurosat_set1': get_eurosat,
				   'oxfordpets_set1':get_oxfordpets}
 