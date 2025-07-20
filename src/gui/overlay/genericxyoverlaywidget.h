#ifndef GENERICXYOVERLAYWIDGET_H
#define GENERICXYOVERLAYWIDGET_H

#include <QSpinBox>
#include <QComboBox>
#include <QLineEdit>
#include <QToolButton>
#include <QFormLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTableWidget>
#include <QSplitter>
#include <QTextEdit>
#include <QCheckBox>
#include <memory>

#include <data/analysis/ft.h>
#include <data/experiment/overlaytypes.h>
#include <data/processing/parsers/genericxyparser.h>
#include <data/storage/settingsstorage.h>
#include "overlaytypespecificwidget.h"

/**
 * @brief Type-specific widget for GenericXY overlays
 * 
 * This widget handles all GenericXY-specific functionality within the
 * UnifiedOverlayWidget architecture. It provides file selection, format
 * detection, column mapping, and preview capabilities for generic XY data files.
 * 
 * Three-Tier UI Architecture:
 * 1. Source File Config: File selection, format detection, preview
 * 2. Source File Settings: Column mapping, parsing options, header handling  
 * 3. Type-Specific Settings: Data filtering, scaling (future)
 */
class GenericXYOverlayWidget : public OverlayTypeSpecificWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit GenericXYOverlayWidget(const Ft &currentFt, QWidget *parent = nullptr);
    ~GenericXYOverlayWidget();

    // OverlayTypeSpecificWidget interface
    void setupForCreation() override;
    void setupForSettings(std::shared_ptr<OverlayBase> overlay) override;
    
    std::shared_ptr<OverlayBase> createOverlay() override;
    void applyToOverlay(std::shared_ptr<OverlayBase> overlay) const override;
    
    bool validateSettings(QString &errorMessage) const override;
    bool isDataValid() const override;
    
    // Source file management
    bool hasValidSourceFile() const override;
    QString getSourceFilePath() const override;
    void setSourceFilePath(const QString &path) override;
    bool validateSourceFile(QString &errorMessage) override;
    
    void resetToDefaults() override;
    
    // Settings state capture for preview sync tracking
    QHash<QString, QVariant> getSettingsHash() const override;
    
    // Operation declaration interface
    QVector<OperationCapability> getSupportedOperations() const override;
    bool supportsBackgroundOperation(OperationCapability::Type type) const override;
    std::shared_ptr<OverlayOperation> createOperation(OperationCapability::Type type,
                                                     std::shared_ptr<OverlayBase> overlay = nullptr) const override;

    // Three-tier architecture interface
    QWidget* getSourceFileConfigWidget() override;
    QWidget* getSourceFileSettingsWidget() override;
    QWidget* getOverlaySettingsWidget() override;

signals:
    void fileChanged(); // Emitted when source file changes

public slots:
    void onFileSelected();
    void onFormatDetectionRequested();
    void onParseSettingsChanged();
    void updatePreview();

private slots:
    void onFileDialogRequested();
    void onDelimiterChanged();
    void onHeaderLinesChanged();
    void onColumnSelectionChanged();
    void onAutoDetectClicked();

private:
    // UI Setup
    void setupUI();
    void setupConnections();
    void createSourceFileConfigUI();
    void createSourceFileSettingsUI();
    void createTypeSpecificSettingsUI();
    
    // File handling
    void loadAndAnalyzeFile();
    void detectFileFormat();
    void updateColumnSelectors();
    void parseAndPreview();
    
    // Settings management
    void loadSettings();
    void saveSettings() override;
    
    // Data validation
    bool validateFileExists() const;
    bool validateColumns() const;
    bool validateDataRange() const;
    
    // Helper methods
    QString getDelimiterDisplayName(const QString &delimiter) const;
    void populateDelimiterComboBox();
    void updateDataStatistics();
    GenericXYParser* getParser() const;
    
    // Core data
    GenericXYData d_parsedData;
    QString d_sourceFilePath;
    bool d_dataValid;
    
    // Tier 1: Source File Config widgets
    QWidget *p_sourceFileConfigWidget;
    QLineEdit *p_filePathEdit;
    QPushButton *p_browseButton;
    QPushButton *p_autoDetectButton;
    QLabel *p_fileStatusLabel;
    
    // Tier 2: Source File Settings widgets  
    QWidget *p_sourceFileSettingsWidget;
    QComboBox *p_delimiterCombo;
    QSpinBox *p_headerLinesSpinBox;
    QComboBox *p_xColumnCombo;
    QComboBox *p_yColumnCombo;
    QCheckBox *p_enableFilteringCheckBox;
    QLineEdit *p_xMinEdit;
    QLineEdit *p_xMaxEdit;
    QLabel *p_dataStatsLabel;
    
    // Tier 3: Type-specific settings - Preview display
    QWidget *p_typeSpecificWidget;
    QTableWidget *p_previewTable;
    
    // State tracking
    bool d_settingsLoaded;
    bool d_fileAnalyzed;
    QStringList d_detectedColumnNames;
};

#endif // GENERICXYOVERLAYWIDGET_H