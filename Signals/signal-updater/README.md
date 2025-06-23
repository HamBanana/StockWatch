# Signal Updater Tool

An intelligent Node.js application for updating stock surge detection signals with new pattern data.

## Overview

This tool efficiently merges update data (like `full-update.json`) into the main `signals.json` configuration file. It handles complex nested structures, validates data integrity, and provides comprehensive reporting.

## Features

- **Intelligent Merging**: Doesn't just append data - intelligently updates existing structures
- **Automatic Backups**: Creates timestamped backups before in-place updates
- **Data Validation**: Ensures weights sum correctly and data integrity is maintained
- **Comprehensive Updates**: Handles all types of updates including:
  - Category and subcategory weights
  - New signal combinations
  - Sector-specific multipliers
  - Technical indicators
  - Volume patterns
  - Decay patterns
  - Market cap tiers
  - Alert thresholds
  - And more...
- **Progress Tracking**: Real-time updates on what's being modified
- **Summary Reports**: Detailed report of all changes made

## Installation

1. Ensure you have Node.js 14+ installed
2. Place the `signal-updater.js` file in your project directory
3. (Optional) Use the provided `package.json` for npm scripts

## Usage

### Command Line

```bash
# Basic usage - updates signals.json in place
node signal-updater.js signals.json full-update.json

# Save to a different file
node signal-updater.js signals.json full-update.json signals-updated.json

# Using npm scripts (if package.json is present)
npm run update              # Updates in place
npm run update-save         # Saves to signals-updated.json
```

### As a Module

```javascript
const SignalUpdater = require('./signal-updater');

async function updateSignals() {
    const updater = new SignalUpdater();
    
    await updater.loadFiles('signals.json', 'full-update.json');
    updater.applyUpdates();
    updater.validateUpdate();
    await updater.saveUpdatedSignals('signals-updated.json');
}
```

## Update File Format

The update file (like `full-update.json`) should contain sections such as:

```json
{
  "meta_update": {
    "description": "Update description",
    "analysis_date": "2025-06-22"
  },
  "category_weight_adjustments": {
    "category_name": {
      "current": 0.10,
      "recommended": 0.15
    }
  },
  "subcategory_updates": {
    "category_name": {
      "new_subcategory": {
        "w": 0.25,
        "t": [80, 90, 100]
      }
    }
  },
  "new_signal_combinations": {
    "combination_name": {
      "cat": ["cat1", "cat2"],
      "scores": [80, 90],
      "multiplier": 2.0
    }
  }
}
```

## Output

The tool provides:

1. **Console Output**: Step-by-step progress of updates being applied
2. **Backup File**: `signals_backup_YYYY-MM-DD_HH-MM-SS.json`
3. **Updated File**: Either in-place or to specified output path
4. **Summary Report**: Statistics on what was updated

### Example Output

```
🚀 Signal Update Tool v1.0

📁 Backup created: signals_backup_2025-06-22_14-30-45.json
✓ Loaded signals.json and update file successfully

📊 Applying updates...

• Updating category weights...
  - ipo: 0.10 → 0.15
  - micro: 0.08 → 0.12
  - volume: 0.22 → 0.25

• Adding new signal combinations...
  - Added: ipoExplosion
  - Added: esgMomentumLowFloat

✓ All validation checks passed

✅ Updated signals saved to: signals.json

📋 Update Summary Report
==================================================
Categories Updated: 3
New Subcategories: 12
New Signal Combinations: 4
Sectors Updated: 3
New Filters: 5
New Technical Indicators: 3
==================================================

✨ Update process completed successfully!
```

## How It Works

1. **Loading**: Reads both JSON files into memory
2. **Deep Merge**: Uses recursive merging to update nested structures
3. **Special Handling**: Recognizes patterns like "recommended" weights and applies them
4. **Validation**: Ensures mathematical constraints (weights sum to 1.0)
5. **Backup**: Creates timestamped backup before modifying original
6. **Save**: Writes formatted JSON with proper indentation

## Error Handling

The tool includes comprehensive error handling for:
- File not found
- Invalid JSON format
- Write permission issues
- Validation failures

## Best Practices

1. Always review the update file before applying
2. Keep backups (automatic, but also periodic manual backups)
3. Validate the output signals in your testing environment
4. Use version control to track changes over time

## Extending the Tool

To handle new update patterns:

1. Add a new section in the `applyUpdates()` method
2. Use the `deepMerge()` function for nested updates
3. Add validation logic in `validateUpdate()` if needed
4. Update the report generation in `generateUpdateReport()`

## Support

For issues or questions about the update patterns, refer to the surge analysis documents or create a new surge document for investigation.