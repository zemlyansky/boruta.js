// const { RandomForestRegression } = require('random-forest')
const { RandomForestRegressor } = require('random-forest')
const fs = require('fs')
const parse = require('csv-parse/lib/sync')

const boruta = require('.') // Change to require('boruta') if installed from npm

// Load data
const file = fs.readFileSync('./friedman1.csv', 'utf8')
const records = parse(file, { columns: false })
const features = records.shift().slice(1, -1)
const X = records.map(r => r.slice(1, -1))
const y = records.map(r => r[r.length - 1])

// Leave the first sample for testing if models work
const Xt = X.shift()
const yt = y.shift()

// Train a random forest
const rf = new RandomForestRegressor({
  maxDepth: 20,
  nEstimators: 50
})
rf.train(X, y)
console.log('Random forest pred:', rf.predict(Xt))
console.log('True Y:', yt)

const bor = boruta(X, y, { names: features })
console.log(bor)

