// const { RandomForestRegression } = require('random-forest')
const { RandomForestRegressor } = require('random-forest')
const fs = require('fs')
const parse = require('csv-parse/lib/sync')
const make = require('mkdata')
const wb = require('@gribnoysup/wunderbar')
const imp = require('importance')

const boruta = require('.') // Change to require('boruta') if installed from npm

// Load data
/*
const file = fs.readFileSync('./friedman1.csv', 'utf8')
const records = parse(file, { columns: false })
const features = records.shift().slice(1, -1)
const X = records.map(r => r.slice(1, -1))
const y = records.map(r => r[r.length - 1])

// Leave the first sample for testing if models work
const Xt = X.shift()
const yt = y.shift()
*/

const [X, y, f1] = make.friedman1({ nSamples: 1000 })
const names = X[0].map((_, i) => 'x' + (i + 1))

const trueModel = {
  predict: X => X.map(x => f1(x))
}

console.log('\nTrue model:')
let tmImp = imp(trueModel, X, y, { kind: 'mse' })
console.log(wb(tmImp, { min: 0, max: 20, length: 50, randomColorOptions: { hue: 'blue', luminosity: 'dark', seed: 0 }}).chart)
console.log(tmImp)

// console.log(trueModel.predict(Xtrain).slice(0, 10), ytrain.slice(0, 10))

// Train a random forest
const rf = new RandomForestRegressor({
  maxDepth: 10,
  nEstimators: 50
})

rf.train(X, y)
// console.log('Random forest pred:', rf.predict(Xt))
// console.log('True Y:', yt)


console.log('\nRandom forest:')
let rfImp = imp(rf, X, y, { kind: 'mse' })
console.log(wb(rfImp, { min: 0, max: 20, length: 50, randomColorOptions: { hue: 'blue', luminosity: 'dark', seed: 0 }}).chart)
console.log(rfImp)

console.log('\nStarting boruta:')
const bor = boruta(X, y, { names })
// const { chart, legend } = wb([1,2,3], { min: 0, length: 50})
// console.log(bor.finalDecision)
// process.exit()

