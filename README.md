# Boruta

Feature selection is a process of filtering variables with some method or criteria ([Wiki](https://en.wikipedia.org/wiki/Feature_selection)).
It often improves a machine learning model performance and helps with data exploration.
Boruta [1] is a feature selection method that identifies *all-relevant* variables, instead of just selecting a minimal subset.
Boruta.js is almost line-by-line port of R's package [Boruta](https://cran.r-project.org/web/packages/Boruta/) to JavaScript.
It depends on the [random-forest](https://www.npmjs.com/package/random-forest) package, but can be used with other models as well.

### Example
```javascript
// Load boruta
const boruta = require('boruta')

// Generate synthetic data
const make = require('mkdata')
const [X, y] = make.friedman1({ nSamples: 1000 })

// Run boruta
const bor = boruta(X, y)

// Print results
console.log(bor.finalDecision)
```

Results:
```javascript
{
  '0': 'Confirmed',
  '1': 'Confirmed',
  '2': 'Rejected',
  '3': 'Confirmed',
  '4': 'Rejected',
  '5': 'Rejected',
  '6': 'Rejected',
  '7': 'Rejected',
  '8': 'Rejected',
  '9': 'Rejected'
}
```

### Web demo
You can try Boruta in the StatSim app: [https://statsim.com/select/](https://statsim.com/select/).
It visualizes importance scores with final decisions and also suports multiple base models (linear regression, logistic regression, KNN, random forest)

### References
1. [Feature Selection with the Boruta Package](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.7660&rep=rep1&type=pdf) (2010) *Miron B. Kursa, Witold R. Rudnicki*



