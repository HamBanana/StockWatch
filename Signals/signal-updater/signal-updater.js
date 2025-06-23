#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');

class SignalUpdater {
    constructor() {
        this.signals = null;
        this.updates = null;
    }

    async loadFiles(signalsPath, updatePath) {
        try {
            const signalsData = await fs.readFile(signalsPath, 'utf8');
            const updateData = await fs.readFile(updatePath, 'utf8');
            
            this.signals = JSON.parse(signalsData);
            this.updates = JSON.parse(updateData);
            
            console.log('✓ Loaded signals.json and update file successfully');
        } catch (error) {
            console.error('Error loading files:', error.message);
            throw error;
        }
    }

    // Deep merge function that handles arrays and objects intelligently
    deepMerge(target, source, path = '') {
        for (const key in source) {
            if (source.hasOwnProperty(key)) {
                const sourcePath = path ? `${path}.${key}` : key;
                
                if (source[key] === null || source[key] === undefined) {
                    continue;
                }
                
                // Handle special update patterns
                if (key === 'recommended' && path.includes('category_weight_adjustments')) {
                    // This is a weight update recommendation
                    continue;
                }
                
                if (Array.isArray(source[key])) {
                    // For arrays, replace entirely
                    target[key] = [...source[key]];
                } else if (typeof source[key] === 'object' && !Array.isArray(source[key])) {
                    // For objects, merge recursively
                    if (!target[key] || typeof target[key] !== 'object') {
                        target[key] = {};
                    }
                    this.deepMerge(target[key], source[key], sourcePath);
                } else {
                    // For primitives, replace
                    target[key] = source[key];
                }
            }
        }
    }

    applyUpdates() {
        console.log('\n📊 Applying updates...\n');

        // 1. Update category weights
        if (this.updates.category_weight_adjustments) {
            console.log('• Updating category weights...');
            for (const [category, adjustment] of Object.entries(this.updates.category_weight_adjustments)) {
                if (adjustment.recommended && this.signals.signals[category]) {
                    const oldWeight = this.signals.signals[category].w;
                    this.signals.signals[category].w = adjustment.recommended;
                    console.log(`  - ${category}: ${oldWeight} → ${adjustment.recommended}`);
                    
                    // Apply conditional adjustments if specified
                    if (adjustment.conditional_boost) {
                        this.signals.signals[category].conditional_boost = adjustment.conditional_boost;
                    }
                    if (adjustment.conditional_adjustments) {
                        this.signals.signals[category].conditional_adjustments = adjustment.conditional_adjustments;
                    }
                }
            }
        }

        // 2. Update subcategories
        if (this.updates.subcategory_updates) {
            console.log('\n• Updating subcategories...');
            for (const [category, subcategories] of Object.entries(this.updates.subcategory_updates)) {
                if (this.signals.signals[category]) {
                    for (const [subcat, data] of Object.entries(subcategories)) {
                        if (!this.signals.signals[category][subcat]) {
                            console.log(`  - Adding new subcategory: ${category}.${subcat}`);
                        } else {
                            console.log(`  - Updating subcategory: ${category}.${subcat}`);
                        }
                        this.signals.signals[category][subcat] = data;
                    }
                }
            }
        }

        // 3. Add new signal combinations
        if (this.updates.new_signal_combinations) {
            console.log('\n• Adding new signal combinations...');
            if (!this.signals.confirm) {
                this.signals.confirm = {};
            }
            for (const [name, combo] of Object.entries(this.updates.new_signal_combinations)) {
                this.signals.confirm[name] = combo;
                console.log(`  - Added: ${name}`);
            }
        }

        // 4. Update sector-specific multipliers
        if (this.updates.sector_specific_updates) {
            console.log('\n• Updating sector multipliers...');
            for (const [sector, updates] of Object.entries(this.updates.sector_specific_updates)) {
                if (!this.signals.sectors[sector]) {
                    this.signals.sectors[sector] = {};
                }
                this.deepMerge(this.signals.sectors[sector], updates);
                console.log(`  - Updated: ${sector}`);
            }
        }

        // 5. Update float thresholds
        if (this.updates.float_threshold_updates) {
            console.log('\n• Updating float thresholds...');
            if (!this.signals.floatTiers) {
                this.signals.floatTiers = {};
            }
            this.deepMerge(this.signals.floatTiers, this.updates.float_threshold_updates);
        }

        // 6. Update filters
        if (this.updates.enhanced_filters) {
            console.log('\n• Updating filters...');
            this.deepMerge(this.signals.filters, this.updates.enhanced_filters);
        }

        // 7. Update decay patterns
        if (this.updates.decay_pattern_updates) {
            console.log('\n• Updating decay patterns...');
            for (const [category, patterns] of Object.entries(this.updates.decay_pattern_updates)) {
                if (!this.signals.decay[category]) {
                    this.signals.decay[category] = {};
                }
                // For decay patterns, we need special handling
                if (patterns.aggressive || patterns.standard || patterns.explosive || patterns.sustained) {
                    this.signals.decay[category].patterns = patterns;
                } else {
                    this.deepMerge(this.signals.decay[category], patterns);
                }
            }
        }

        // 8. Update alert thresholds
        if (this.updates.alert_threshold_adjustments) {
            console.log('\n• Adding new alert thresholds...');
            if (!this.signals.alerts.conditional) {
                this.signals.alerts.conditional = {};
            }
            this.deepMerge(this.signals.alerts.conditional, this.updates.alert_threshold_adjustments);
        }

        // 9. Add technical indicators
        if (this.updates.technical_indicator_additions) {
            console.log('\n• Adding technical indicators...');
            if (!this.signals.technicalIndicators) {
                this.signals.technicalIndicators = {};
            }
            this.deepMerge(this.signals.technicalIndicators, this.updates.technical_indicator_additions);
        }

        // 10. Add volume patterns
        if (this.updates.volume_pattern_recognition) {
            console.log('\n• Adding volume pattern recognition...');
            if (!this.signals.volumePatterns) {
                this.signals.volumePatterns = {};
            }
            this.deepMerge(this.signals.volumePatterns, this.updates.volume_pattern_recognition);
        }

        // 11. Update validation with DGNX
        if (this.updates.comprehensive_validation_entry) {
            console.log('\n• Adding validation entries...');
            this.deepMerge(this.signals.validation.documented, this.updates.comprehensive_validation_entry);
        }

        // 12. Update market cap tiers
        if (this.updates.market_cap_tier_adjustments) {
            console.log('\n• Updating market cap tiers...');
            if (!this.signals.marketCapTiers) {
                this.signals.marketCapTiers = {};
            }
            this.deepMerge(this.signals.marketCapTiers, this.updates.market_cap_tier_adjustments);
        }

        // 13. Add news catalyst scoring
        if (this.updates.news_catalyst_scoring) {
            console.log('\n• Adding news catalyst scoring...');
            if (!this.signals.catalystScoring) {
                this.signals.catalystScoring = {};
            }
            this.deepMerge(this.signals.catalystScoring, this.updates.news_catalyst_scoring);
        }

        // 14. Update meta information
        if (this.updates.meta_update) {
            console.log('\n• Updating meta information...');
            const today = new Date().toISOString().split('T')[0];
            this.signals.meta.updated = today;
            this.signals.meta.lastUpdate = {
                date: today,
                source: 'DGNX surge analysis',
                version: this.signals.meta.v
            };
        }

        console.log('\n✅ All updates applied successfully!');
    }

    validateUpdate() {
        console.log('\n🔍 Validating updated signals...\n');
        
        let issues = [];

        // Check that all weights sum to approximately 1
        let totalWeight = 0;
        for (const [category, data] of Object.entries(this.signals.signals)) {
            if (data.w) {
                totalWeight += data.w;
            }
        }
        
        if (Math.abs(totalWeight - 1.0) > 0.01) {
            issues.push(`Category weights sum to ${totalWeight.toFixed(3)} instead of 1.0`);
        }

        // Check subcategory weights
        for (const [category, data] of Object.entries(this.signals.signals)) {
            let subWeight = 0;
            for (const [key, value] of Object.entries(data)) {
                if (key !== 'w' && value.w) {
                    subWeight += value.w;
                }
            }
            if (subWeight > 0 && Math.abs(subWeight - 1.0) > 0.01) {
                issues.push(`${category} subcategory weights sum to ${subWeight.toFixed(3)}`);
            }
        }

        if (issues.length > 0) {
            console.log('⚠️  Validation warnings:');
            issues.forEach(issue => console.log(`  - ${issue}`));
        } else {
            console.log('✓ All validation checks passed');
        }
    }

    async saveUpdatedSignals(outputPath) {
        try {
            const updatedJson = JSON.stringify(this.signals, null, 2);
            await fs.writeFile(outputPath, updatedJson);
            console.log(`\n✅ Updated signals saved to: ${outputPath}`);
        } catch (error) {
            console.error('Error saving updated signals:', error.message);
            throw error;
        }
    }

    async createBackup(signalsPath) {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            const backupPath = signalsPath.replace('.json', `_backup_${timestamp}.json`);
            await fs.copyFile(signalsPath, backupPath);
            console.log(`📁 Backup created: ${backupPath}`);
            return backupPath;
        } catch (error) {
            console.error('Error creating backup:', error.message);
            throw error;
        }
    }

    generateUpdateReport() {
        console.log('\n📋 Update Summary Report\n');
        console.log('=' * 50);
        
        const report = {
            categoriesUpdated: Object.keys(this.updates.category_weight_adjustments || {}).length,
            newSubcategories: 0,
            newCombinations: Object.keys(this.updates.new_signal_combinations || {}).length,
            sectorsUpdated: Object.keys(this.updates.sector_specific_updates || {}).length,
            newFilters: Object.keys(this.updates.enhanced_filters || {}).length,
            newIndicators: Object.keys(this.updates.technical_indicator_additions || {}).length,
        };

        // Count new subcategories
        if (this.updates.subcategory_updates) {
            for (const subcats of Object.values(this.updates.subcategory_updates)) {
                report.newSubcategories += Object.keys(subcats).length;
            }
        }

        console.log(`Categories Updated: ${report.categoriesUpdated}`);
        console.log(`New Subcategories: ${report.newSubcategories}`);
        console.log(`New Signal Combinations: ${report.newCombinations}`);
        console.log(`Sectors Updated: ${report.sectorsUpdated}`);
        console.log(`New Filters: ${report.newFilters}`);
        console.log(`New Technical Indicators: ${report.newIndicators}`);
        console.log('=' * 50);
    }
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);
    
    if (args.length < 2) {
        console.log('Usage: node signal-updater.js <signals.json> <update.json> [output.json]');
        console.log('\nExample:');
        console.log('  node signal-updater.js signals.json full-update.json signals-updated.json');
        console.log('\nIf output path is not specified, signals.json will be updated in place.');
        process.exit(1);
    }

    const signalsPath = args[0];
    const updatePath = args[1];
    const outputPath = args[2] || signalsPath;

    const updater = new SignalUpdater();

    try {
        console.log('🚀 Signal Update Tool v1.0\n');
        
        // Create backup if updating in place
        if (outputPath === signalsPath) {
            await updater.createBackup(signalsPath);
        }

        // Load files
        await updater.loadFiles(signalsPath, updatePath);

        // Apply updates
        updater.applyUpdates();

        // Validate
        updater.validateUpdate();

        // Save
        await updater.saveUpdatedSignals(outputPath);

        // Generate report
        updater.generateUpdateReport();

        console.log('\n✨ Update process completed successfully!');

    } catch (error) {
        console.error('\n❌ Update failed:', error.message);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}

// Export for use as module
module.exports = SignalUpdater;