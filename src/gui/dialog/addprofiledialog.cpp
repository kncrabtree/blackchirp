#include "addprofiledialog.h"

#include <QComboBox>
#include <QLineEdit>
#include <QLabel>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QMessageBox>
#include <QFileDialog>
#include <QMetaEnum>
#include <QSettings>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>
#include <data/storage/settingsstorage.h>
#include <hardware/core/hardwareregistry.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <hardware/core/communication/communicationprotocol.h>
#include <gui/style/themecolors.h>
#include <gui/widget/hwsettingswidget.h>
#include <gui/widget/customprotocolwidget.h>

AddProfileDialog::AddProfileDialog(const QString &hardwareType, QWidget *parent)
    : QDialog(parent), d_hardwareType(hardwareType)
{
    setWindowTitle("Add " + hardwareType + " Profile");
    setModal(true);
    setMinimumWidth(520);

    auto *layout = new QVBoxLayout(this);

    // Get available drivers
    QStringList implementations = HardwareRegistry::instance().getImplementations(hardwareType);
    implementations.sort();

    auto *formLayout = new QFormLayout();

    if (implementations.isEmpty()) {
        layout->addWidget(new QLabel("No drivers available"));
        p_implementationCombo = new QComboBox();
        p_protocolCombo = new QComboBox();
        p_protocolLabel = new QLabel("Protocol:");
        p_labelEdit = new QLineEdit();
        p_settingsContainer = new QWidget();
        p_validationLabel = new QLabel();
        p_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        connect(p_buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(p_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
        layout->addWidget(p_buttonBox);
        return;
    }

    // Driver combo
    p_implementationCombo = new QComboBox();
    p_implementationCombo->addItems(implementations);
    formLayout->addRow("Driver:", p_implementationCombo);

    // Protocol combo (shown only when multiple protocols are supported)
    p_protocolLabel = new QLabel("Protocol:");
    p_protocolCombo = new QComboBox();
    formLayout->addRow(p_protocolLabel, p_protocolCombo);

    // Label input
    p_labelEdit = new QLineEdit();
    p_labelEdit->setPlaceholderText("Enter unique label for this profile");
    QString defaultLabel = HardwareProfileManager::instance().generateDefaultLabel(hardwareType);
    p_labelEdit->setText(defaultLabel);
    formLayout->addRow("Label:", p_labelEdit);

    layout->addLayout(formLayout);

    // Container that holds the current HwSettingsWidget (swapped on impl change)
    p_settingsContainer = new QWidget(this);
    auto *containerLayout = new QVBoxLayout(p_settingsContainer);
    containerLayout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(p_settingsContainer);

    // Container that holds the CustomProtocolWidget (swapped on impl change, shown only for Custom protocol)
    p_customCommContainer = new QWidget(this);
    auto *customCommLayout = new QVBoxLayout(p_customCommContainer);
    customCommLayout->setContentsMargins(0, 0, 0, 0);
    p_customCommContainer->hide();
    layout->addWidget(p_customCommContainer);

    // Validation label
    p_validationLabel = new QLabel();
    p_validationLabel->setStyleSheet(QString("QLabel { color: %1; }")
        .arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
    p_validationLabel->hide();
    layout->addWidget(p_validationLabel);

    // Button box
    p_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(p_buttonBox);

    connect(p_buttonBox, &QDialogButtonBox::accepted, this, &AddProfileDialog::accept);
    connect(p_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // Connect implementation combo changes
    connect(p_implementationCombo, &QComboBox::currentTextChanged,
            this, &AddProfileDialog::updateProtocolCombo);
    connect(p_implementationCombo, &QComboBox::currentTextChanged,
            this, &AddProfileDialog::updateSettingsDefs);
    connect(p_implementationCombo, &QComboBox::currentTextChanged,
            this, &AddProfileDialog::updateCustomCommWidget);

    // Show/hide custom comm widget when the protocol selection changes
    connect(p_protocolCombo, &QComboBox::currentIndexChanged, this, [this](int) {
        updateCustomCommWidget(p_implementationCombo->currentText());
    });

    // Connect label validation
    connect(p_labelEdit, &QLineEdit::textChanged,
            this, &AddProfileDialog::validateLabel);

    // Initialize with current selection
    updateProtocolCombo(p_implementationCombo->currentText());
    updateSettingsDefs(p_implementationCombo->currentText());
    updateCustomCommWidget(p_implementationCombo->currentText());

    // Trigger initial validation
    p_labelEdit->textChanged(p_labelEdit->text());

    adjustSize();
}

QString AddProfileDialog::selectedImplementation() const
{
    return p_implementationCombo->currentText();
}

QString AddProfileDialog::profileLabel() const
{
    return p_labelEdit->text().trimmed();
}

QString AddProfileDialog::pythonScriptPath() const
{
    return d_pythonScriptPath;
}

void AddProfileDialog::updateProtocolCombo(const QString &impl)
{
    auto protocols = HardwareRegistry::instance().getSupportedProtocols(d_hardwareType, impl);
    p_protocolCombo->clear();
    auto commEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
    for (auto p : protocols)
        p_protocolCombo->addItem(QString(commEnum.valueToKey(static_cast<int>(p))),
                                 static_cast<int>(p));
    bool multiProtocol = protocols.size() > 1;
    p_protocolLabel->setVisible(multiProtocol);
    p_protocolCombo->setVisible(multiProtocol);
}

void AddProfileDialog::updateSettingsDefs(const QString &impl)
{
    // Remove and delete the old settings widget if present
    delete p_settingsWidget;
    p_settingsWidget = nullptr;

    p_settingsWidget = new HwSettingsWidget(d_hardwareType, impl,
                                            HwSettingsMode::Create,
                                            /*storageKey=*/{}, this);
    p_settingsContainer->layout()->addWidget(p_settingsWidget);
}

void AddProfileDialog::updateCustomCommWidget(const QString &impl)
{
    delete p_customCommWidget;
    p_customCommWidget = nullptr;

    auto selectedProtocol = static_cast<CommunicationProtocol::CommType>(
        p_protocolCombo->currentData().toInt());

    auto protocols = HardwareRegistry::instance().getSupportedProtocols(d_hardwareType, impl);
    bool supportsCustom = protocols.contains(CommunicationProtocol::Custom);

    if (supportsCustom && selectedProtocol == CommunicationProtocol::Custom) {
        p_customCommWidget = new CustomProtocolWidget(d_hardwareType, impl, this);
        p_customCommContainer->layout()->addWidget(p_customCommWidget);
        p_customCommContainer->show();
    } else {
        p_customCommContainer->hide();
    }
}

void AddProfileDialog::validateLabel(const QString &text)
{
    auto &profileManager = HardwareProfileManager::instance();
    auto validationError = profileManager.validateLabel(text);
    if (validationError != HardwareProfileManager::Valid) {
        QString errorMsg;
        switch (validationError) {
            case HardwareProfileManager::Empty:
                errorMsg = "Label cannot be empty";
                break;
            case HardwareProfileManager::TooLong:
                errorMsg = "Label too long (max 64 characters)";
                break;
            case HardwareProfileManager::InvalidCharacters:
                errorMsg = "Label contains invalid characters";
                break;
            case HardwareProfileManager::StartsWithNumber:
                errorMsg = "Label cannot start with a number";
                break;
            case HardwareProfileManager::StartsWithUnderscore:
                errorMsg = "Label cannot start with underscore";
                break;
            case HardwareProfileManager::ContainsDots:
                errorMsg = "Label cannot contain dots";
                break;
            default:
                errorMsg = "Invalid label";
                break;
        }
        p_validationLabel->setText(errorMsg);
        p_validationLabel->show();
        p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
    } else if (!profileManager.isLabelAvailable(d_hardwareType, text)) {
        p_validationLabel->setText("Label already exists for this hardware type");
        p_validationLabel->show();
        p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
    } else {
        p_validationLabel->hide();
        p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
    }
}

void AddProfileDialog::accept()
{
    QString implementation = p_implementationCombo->currentText();
    QString label = p_labelEdit->text().trimmed();
    QString settingsKey = BC::Key::hwKey(d_hardwareType, label);

    auto selectedProtocol = static_cast<CommunicationProtocol::CommType>(
        p_protocolCombo->currentData().toInt());

    // Write the selected communication protocol to settings before hardware
    // object construction so the constructor finds the correct value.
    {
        QSettings s;
        s.beginGroup(settingsKey);
        s.setValue(BC::Key::HW::commType, static_cast<int>(selectedProtocol));
        s.endGroup();
        s.sync();
    }

    // Write all hardware settings (scalars + arrays) from the widget
    if (p_settingsWidget)
        p_settingsWidget->saveToStorage(settingsKey);

    // Write custom communication parameters if a Custom protocol widget is present
    if (p_customCommWidget && selectedProtocol == CommunicationProtocol::Custom)
        p_customCommWidget->saveToStorage(settingsKey);

    // Create the profile
    auto &profileManager = HardwareProfileManager::instance();
    QString actualLabel = profileManager.createHardwareProfile(d_hardwareType, implementation, label);

    if (actualLabel.isEmpty()) {
        QMessageBox::warning(this, "Add Profile", "Failed to create profile. Please try again.");
        return;
    }

    if (implementation.startsWith(QStringLiteral("Python")))
        offerPythonTemplate();

    QDialog::accept();
}

void AddProfileDialog::offerPythonTemplate()
{
    QString implementation = p_implementationCombo->currentText();

    // Derive template filename: "PythonAwg" -> "python_awg_template.py"
    QString typePart = implementation.mid(6); // Remove "Python" prefix
    QString templateFilename = QStringLiteral("python_%1_template.py").arg(typePart.toLower());

    // Search for template using same paths as findHostScript()
    QString appDir = QCoreApplication::applicationDirPath();
    QString templatePath;
    QStringList searchPaths = {
        appDir + "/" + templateFilename,
        appDir + "/../share/blackchirp/" + templateFilename
    };
    for (const auto &path : searchPaths) {
        if (QFile::exists(path)) {
            templatePath = path;
            break;
        }
    }

    if (templatePath.isEmpty())
        return;

    auto result = QMessageBox::question(
        this, tr("Python Template Script"),
        tr("<b>Python hardware scripts run with full system access.</b> "
           "Scripts can access files, network resources, and hardware devices "
           "with the same permissions as Blackchirp. Only use scripts from "
           "sources you trust."
           "<br><br>"
           "Would you like to create a copy of the template script to customize?"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);

    if (result == QMessageBox::Yes) {
        SettingsStorage ss;
        QString initialDir = ss.get(BC::Key::savePath, QString(""));
        QString suggestedName = QStringLiteral("my_%1.py").arg(typePart.toLower());
        if (!initialDir.isEmpty())
            suggestedName = QDir(initialDir).filePath(suggestedName);
        QString savePath = QFileDialog::getSaveFileName(
            this, tr("Save Python Script"),
            suggestedName,
            tr("Python Scripts (*.py)"));

        if (!savePath.isEmpty()) {
            if (QFile::exists(savePath))
                QFile::remove(savePath);

            if (QFile::copy(templatePath, savePath)) {
                QFile::setPermissions(savePath,
                    QFile::permissions(savePath) | QFileDevice::WriteOwner);
                d_pythonScriptPath = savePath;
            } else {
                QMessageBox::warning(this, tr("Python Template Script"),
                    tr("Failed to copy template script to the selected location."));
            }
        }
    }
}
