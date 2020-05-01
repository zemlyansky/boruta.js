// const rf = require('random-forest')
const { RandomForest } = require('/home/anton/projects/random-forest/index.js')
const pbinom = require('binomial-cdf')

function getImportanceRFwasm (X, y, opts) {
  const rf = new RandomForest({
    maxDepth: opts.maxDepth || 10,
    nEstimators: opts.nEstimators || 50,
    type: opts.type || 'auto'
  })
  rf.train(X, y)
  const imp = rf.getFeatureImportances(X, y, { n: opts.nRepeats, means: true, verbose: false })
  return imp
}

const defaults = {
  holdHistory: true,
  pValue: 0.01,
  verbose: true,
  maxRuns: 50,
  getImportance: getImportanceRFwasm
}

module.exports = function boruta (X, y, opts = {}) {
  const options = Object.assign({}, defaults, opts)
  const log = options.verbose ? console.log : () => {}
  log('Starting Boruta (p: %.2f, n: %d, maxRuns: %d)...', options.pValue, options.nRepeats, options.maxRuns)

  if ((typeof X === 'object') && (!Array.isArray(X)) && (typeof y === 'string')) {
    // Assume dataframe (object) here and convert to 2D array
    options.names = Object.keys(X)
    const Xtmp = []
    if (Array.isArray(X[options.names[0]])) {
      const length = X[options.names[0]].length
      for (let i = 0; i < length; i++) {
        Xtmp.push(options.names.map(name => X[name][i]))
      }
      X = Xtmp
    } else {
      throw new Error('Unknown data type')
    }
  }

  if (!Array.isArray(X) || !Array.isArray(y)) {
    throw new Error('Inputs are not arrays')
  // We probably don't need to check NaN here because different models can deal with them in their own ways
  // } else if (X.flat().some(v => isNaN(v)) || y.some(v => isNaN(v))) {
  //  throw new Error('NaNs in the data')
  } else if (options.maxRuns < 11) {
    throw new Error('maxRuns must be greater than 10')
  }

  // Useful constants
  const nAtt = X[0].length
  const nObjects = X.length
  const attNames = options.names || X[0].map((_, i) => i + '')
  const confLevels = {
    '0': 'Tentative',
    '1': 'Confirmed',
    '-1': 'Rejected'
  }

  // Initiate state
  let decReg = Array(nAtt).fill(0) // <- Tentative
  let hitReg = Array(nAtt).fill(0) // {}; attNames.forEach(name => { hitReg[name] = 0})
  let impHistory = []
  let runs = 0

  function addShadowsAndGetImp (decReg, runs) {
    // xSha is going to be a data frame with shadow attributes; time to init it.

    let Xfilt = X.map(x => x.filter((_, i) => decReg[i] !== -1))
    // let namesXfilt = attNames.filter((_, i) => decReg[i] !== -1)
    let Xsha = JSON.parse(JSON.stringify(Xfilt))

    while (Xsha[0].length < 5) {
      // There must be at least 5 random attributes.
      // log('Len: %d, Growing shadow matrix', Xsha[0].length)
      Xsha = Xsha.map(x => x.concat(x))
      // names = neams.concat(names)
    }

    // Now, we permute values in each attribute
    const nSha = Xsha[0].length

    for (let row = Xsha.length - 1; row > 0; row--) {
      for (let col = nSha - 1; col > 0; col--) {
        const i = Math.floor(Math.random() * (row + 1))
        ;[Xsha[row][col], Xsha[i][col]] = [Xsha[i][col], Xsha[row][col]]
      }
    }

    const namesXsha = Xsha[0].map((_, i) => 'shadow' + i)

    // Notifying user of the progress
    log(' %s. run of importance source...', runs)

    // Calling importance source
    const Xbind = Xfilt.map((x, i) => x.concat(Xsha[i]))
    const impRaw = options.getImportance(Xbind, y, options)
    // console.log(impRaw)

    if (impRaw.length !== Xfilt[0].length + Xsha[0].length) {
      log('getImp result has a wrong length %d != %d + %d. Please check the given getImportance function', impRaw.length, Xbind[0].length, Xsha[0].length)
      // throw new Error ('getImp result has a wrong length %d != %d. Please check the given getImportance function', impRaw.length, Xbind[0].length + Xsha[0].length)
    }

    // Importance must have Rejected attributes put on place and filled with -Infs
    let imp = new Array(nAtt + nSha).fill(-Infinity)
    // const namesImp = attNames.concat(namesXsha)

    for (let i = 0, j = 0; i < imp.length - 1; i++) {
      if (typeof decReg[i] === 'undefined' || decReg[i] !== -1) {
        imp[i] = impRaw[j]
        j++
      }
    }

    const shaImp = impRaw.slice(nAtt)
    imp = imp.slice(0, nAtt)

    return { imp, shaImp }
  }

  function assignHits (hitReg, curImp) {
    const maxShaImp = Math.max.apply(null, curImp.shaImp)
    const hits = curImp.imp.map(v => v > maxShaImp)
    hits.forEach((v, i) => { hitReg[i] += +v })
    return hitReg
  }

  // Checks whether number of hits is significant
  function doTests (decReg, hitReg, runs) {
    // If attribute is significantly more frequent better than shadowMax, its claimed Confirmed
    const toAccept = hitReg.map((hr, i) => {
      return ((1 - pbinom(hr - 1, runs, 0.5)) < options.pValue) && (decReg[i] === 0)
    })

    // If attribute is significantly more frequent worse than shadowMax, its claimed Rejected (=irrelevant)
    const toReject = hitReg.map((hr, i) => {
      return (pbinom(hr, runs, 0.5) < options.pValue) && (decReg[i] === 0)
    })

    for (let i = 0; i < nAtt; i++) {
      if (toAccept[i]) {
        decReg[i] = 1
      } else if (toReject[i]) {
        decReg[i] = -1
      }
    }

    if (options.verbose) {
      const nAcc = toAccept.reduce((a, v) => a + v, 0)
      const nRej = toReject.reduce((a, v) => a + v, 0)
      const nLeft = decReg.map(v => v === 0).reduce((a, v) => a + v, 0)
      if (nAcc + nRej > 0) {
        console.log('After %s iterations: ', runs)
      }
      if (nAcc > 0) {
        console.log('  (+) confirmed %s attribute%s: %s', nAcc, nAcc === 1 ? '' : 's', attNames.filter((n, i) => toAccept[i]))
      }
      if (nRej > 0) {
        console.log('  (-) rejected %s attribute%s: %s', nRej, nRej === 1 ? '' : 's', attNames.filter((n, i) => toReject[i]))
      }
      if (nAcc + nRej > 0) {
        if (nLeft > 0) {
          console.log('  (?) still have %s attribute%s left\n', nLeft, nLeft === 1 ? '' : 's')
        } else {
          if (nAcc + nRej > 0) {
            console.log('      no more attributes left\n')
          }
        }
      }
    }
    return decReg
  }

  while (decReg.some(v => v === 0) && (runs++ < options.maxRuns)) {
    const curImp = addShadowsAndGetImp(decReg, runs)
    hitReg = assignHits(hitReg, curImp)
    decReg = doTests(decReg, hitReg, runs)

    if (options.holdHistory) {
      impHistory.push({
        imp: curImp.imp,
        shadowMax: Math.max.apply(null, curImp.shaImp),
        shadowMin: Math.min.apply(null, curImp.shaImp),
        shadowMean: curImp.shaImp.reduce((a, v) => a + v / curImp.shaImp.length, 0)
      })
    }
  }

  const decRegObj = {}
  decReg.forEach((v, i) => {
    decRegObj[attNames[i]] = confLevels[v + '']
  })

  log('Final decision:', decRegObj)

  return {
    finalDecision: decRegObj,
    impHistory,
    pValue: options.pValue,
    maxRuns: options.maxRuns,
    impSource: options.getImportance.name
  }
}
