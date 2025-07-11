#ifndef PRESETSAVEDIALOG_H
#define PRESETSAVEDIALOG_H

#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QRadioButton>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>

class CurveAppearancePresetManager;

class PresetSaveDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PresetSaveDialog(const QString &suggestedName, 
                             CurveAppearancePresetManager *presetManager,
                             QWidget *parent = nullptr);

    QString getPresetName() const;
    bool isOverwriteMode() const;

private slots:
    void onModeChanged();
    void onExistingPresetChanged();
    void onNewNameChanged();
    void updateOkButtonState();

private:
    void setupUI();
    void populateExistingPresets();

    // UI components
    QVBoxLayout *p_mainLayout;
    QRadioButton *p_createNewRadio;
    QRadioButton *p_overwriteExistingRadio;
    QButtonGroup *p_modeGroup;
    
    QLineEdit *p_newNameEdit;
    QComboBox *p_existingPresetCombo;
    
    QPushButton *p_okButton;
    QPushButton *p_cancelButton;

    // Data
    QString d_suggestedName;
    CurveAppearancePresetManager *p_presetManager;
};

#endif // PRESETSAVEDIALOG_H