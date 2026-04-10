#include "addprofiledialog.h"

#include <QComboBox>
#include <QLineEdit>
#include <QLabel>
#include <QGroupBox>
#include <QFormLayout>
#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QMessageBox>
#include <QFileDialog>
#include <QMetaEnum>
#include <QSettings>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <limits>
#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>
#include <data/storage/settingsstorage.h>
#include <hardware/core/hardwareregistry.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <hardware/core/communication/communicationprotocol.h>
#include <gui/style/themecolors.h>
#include <gui/widget/scientificspinbox.h>

AddProfileDialog::AddProfileDialog(const QString &hardwareType, QWidget *parent)
    : QDialog(parent), d_hardwareType(hardwareType)
{
    setWindowTitle("Add " + hardwareType + " Profile");
    setModal(true);
    resize(400, 200);

    auto *layout = new QVBoxLayout(this);

    // Get available implementations
    QStringList implementations = HardwareRegistry::instance().getImplementations(hardwareType);
    implementations.sort();

    auto *formLayout = new QFormLayout();

    if (implementations.isEmpty()) {
        layout->addWidget(new QLabel("No implementations available"));
        p_implementationCombo = new QComboBox();
        p_protocolCombo = new QComboBox();
        p_protocolLabel = new QLabel("Protocol:");
        p_labelEdit = new QLineEdit();
        p_requiredParamsGroup = new QGroupBox("Required Settings");
        p_requiredParamsLayout = new QFormLayout(p_requiredParamsGroup);
        p_importantParamsGroup = new QGroupBox("Important Settings");
        p_importantParamsLayout = new QFormLayout(p_importantParamsGroup);
        p_advancedParamsGroup = new QGroupBox("Advanced Settings");
        p_advancedTable = new QTableWidget(0, 2);
        p_validationLabel = new QLabel();
        p_buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        p_buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        connect(p_buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(p_buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
        layout->addWidget(p_buttonBox);
        return;
    }

    // Implementation combo
    p_implementationCombo = new QComboBox();
    p_implementationCombo->addItems(implementations);
    formLayout->addRow("Implementation:", p_implementationCombo);

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

    // Required settings group
    p_requiredParamsGroup = new QGroupBox("Required Settings");
    p_requiredParamsLayout = new QFormLayout(p_requiredParamsGroup);
    p_requiredParamsGroup->hide();
    layout->addWidget(p_requiredParamsGroup);

    // Important settings group
    p_importantParamsGroup = new QGroupBox("Important Settings");
    p_importantParamsLayout = new QFormLayout(p_importantParamsGroup);
    p_importantParamsGroup->hide();
    layout->addWidget(p_importantParamsGroup);

    // Advanced settings group (checkable/collapsible, collapsed by default)
    p_advancedParamsGroup = new QGroupBox("Advanced Settings");
    p_advancedParamsGroup->setCheckable(true);
    p_advancedParamsGroup->setChecked(false);
    auto *advGroupLayout = new QVBoxLayout(p_advancedParamsGroup);
    advGroupLayout->setContentsMargins(4, 4, 4, 4);
    p_advancedTable = new QTableWidget(0, 2);
    p_advancedTable->setHorizontalHeaderLabels({"Setting", "Value"});
    p_advancedTable->horizontalHeader()->setStretchLastSection(true);
    p_advancedTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    p_advancedTable->verticalHeader()->setVisible(false);
    p_advancedTable->setSelectionMode(QAbstractItemView::NoSelection);
    p_advancedTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    p_advancedTable->setMinimumHeight(80);
    p_advancedTable->setMaximumHeight(220);
    advGroupLayout->addWidget(p_advancedTable);
    p_advancedTable->hide();
    p_advancedParamsGroup->hide();
    connect(p_advancedParamsGroup, &QGroupBox::toggled,
            p_advancedTable, &QWidget::setVisible);
    connect(p_advancedParamsGroup, &QGroupBox::toggled,
            this, [this]{ adjustSize(); });
    layout->addWidget(p_advancedParamsGroup);

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

    // Connect label validation
    connect(p_labelEdit, &QLineEdit::textChanged,
            this, &AddProfileDialog::validateLabel);

    // Initialize with current selection
    updateProtocolCombo(p_implementationCombo->currentText());
    updateSettingsDefs(p_implementationCombo->currentText());

    // Trigger initial validation
    p_labelEdit->textChanged(p_labelEdit->text());
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
    // Clear form layouts and table
    while (p_requiredParamsLayout->rowCount() > 0)
        p_requiredParamsLayout->removeRow(0);
    while (p_importantParamsLayout->rowCount() > 0)
        p_importantParamsLayout->removeRow(0);
    p_advancedTable->setRowCount(0);
    d_paramWidgets.clear();

    auto &reg = HardwareRegistry::instance();
    auto settingDefs = reg.getSettingDefs(d_hardwareType, impl);

    // Create a widget for a scalar setting. Int/double/bool/string types are
    // handled. Proper ranges are applied; unlimited ints use INT_MIN/INT_MAX
    // and doubles use ScientificSpinBox (which defaults to ±double::max).
    auto makeWidget = [&](const QString &key, const QString &description,
                           const QVariant &defaultValue,
                           const QVariant &minimum, const QVariant &maximum) -> QWidget* {
        int typeId = defaultValue.userType();
        QWidget *widget = nullptr;
        if (typeId == QMetaType::Int) {
            auto *sb = new QSpinBox();
            sb->setRange(minimum.isValid() ? minimum.toInt() : std::numeric_limits<int>::min(),
                         maximum.isValid() ? maximum.toInt() : std::numeric_limits<int>::max());
            sb->setValue(defaultValue.toInt());
            widget = sb;
        } else if (typeId == QMetaType::UInt) {
            auto *sb = new QSpinBox();
            sb->setRange(minimum.isValid() ? static_cast<int>(minimum.toUInt()) : 0,
                         maximum.isValid() ? static_cast<int>(maximum.toUInt()) : std::numeric_limits<int>::max());
            sb->setValue(static_cast<int>(defaultValue.toUInt()));
            widget = sb;
        } else if (typeId == QMetaType::Double) {
            auto *ssb = new ScientificSpinBox();
            if (minimum.isValid()) ssb->setMinimum(minimum.toDouble());
            if (maximum.isValid()) ssb->setMaximum(maximum.toDouble());
            ssb->setValue(defaultValue.toDouble());
            widget = ssb;
        } else if (typeId == QMetaType::Bool) {
            auto *cb = new QCheckBox();
            cb->setChecked(defaultValue.toBool());
            widget = cb;
        } else {
            auto *le = new QLineEdit();
            le->setText(defaultValue.toString());
            widget = le;
        }
        if (widget) {
            widget->setToolTip(description);
            d_paramWidgets[key] = widget;
        }
        return widget;
    };

    // Add a setting row to the advanced table. Label in col 0 (read-only),
    // widget in col 1 via setCellWidget. Row height auto-fits the widget.
    auto addTableRow = [&](const QString &label, const QString &description,
                            QWidget *widget) {
        int row = p_advancedTable->rowCount();
        p_advancedTable->insertRow(row);
        auto *labelItem = new QTableWidgetItem(label);
        labelItem->setFlags(Qt::ItemIsEnabled);
        labelItem->setToolTip(description);
        p_advancedTable->setItem(row, 0, labelItem);
        if (widget) {
            p_advancedTable->setCellWidget(row, 1, widget);
            p_advancedTable->setRowHeight(row, widget->sizeHint().height());
        }
    };

    if (!settingDefs.isEmpty()) {
        // New priority-grouped UI
        auto arrayDefs = reg.getArraySettingDefs(d_hardwareType, impl);

        bool hasRequired = false, hasImportant = false, hasAdvanced = false;

        for (const auto &s : settingDefs) {
            QWidget *widget = makeWidget(s.key, s.description,
                                         s.defaultValue, s.minimum, s.maximum);
            if (!widget) continue;
            switch (s.priority) {
            case HwSettingPriority::Required:
                p_requiredParamsLayout->addRow(s.label + ":", widget);
                hasRequired = true;
                break;
            case HwSettingPriority::Important:
                p_importantParamsLayout->addRow(s.label + ":", widget);
                hasImportant = true;
                break;
            case HwSettingPriority::Optional:
                addTableRow(s.label, s.description, widget);
                hasAdvanced = true;
                break;
            }
        }

        // Array setting summaries: form row for Required/Important,
        // table row for Optional (read-only count label).
        for (auto it = arrayDefs.cbegin(); it != arrayDefs.cend(); ++it) {
            const auto &arr = it.value();
            QString summary = QString("%1 entries").arg(arr.entries.size());
            switch (arr.priority) {
            case HwSettingPriority::Required: {
                auto *lbl = new QLabel(summary);
                lbl->setToolTip(arr.description);
                p_requiredParamsLayout->addRow(arr.label + ":", lbl);
                hasRequired = true;
                break;
            }
            case HwSettingPriority::Important: {
                auto *lbl = new QLabel(summary);
                lbl->setToolTip(arr.description);
                p_importantParamsLayout->addRow(arr.label + ":", lbl);
                hasImportant = true;
                break;
            }
            case HwSettingPriority::Optional: {
                auto *lbl = new QLabel(summary);
                lbl->setToolTip(arr.description);
                addTableRow(arr.label, arr.description, lbl);
                hasAdvanced = true;
                break;
            }
            }
        }

        p_requiredParamsGroup->setTitle("Required Settings");
        p_requiredParamsGroup->setVisible(hasRequired);
        p_importantParamsGroup->setVisible(hasImportant);
        p_advancedParamsGroup->setVisible(hasAdvanced);
        if (hasAdvanced) {
            p_advancedParamsGroup->setChecked(false);
            p_advancedTable->setVisible(false);
        }
    } else {
        // Fallback: old configParams behavior, shown in required group
        auto params = reg.getConfigParams(d_hardwareType, impl);

        for (const auto &param : params) {
            int typeId = param.defaultValue.userType();
            QWidget *widget = nullptr;
            if (typeId == QMetaType::Int) {
                auto *sb = new QSpinBox();
                sb->setRange(param.minimum.isValid() ? param.minimum.toInt() : std::numeric_limits<int>::min(),
                             param.maximum.isValid() ? param.maximum.toInt() : std::numeric_limits<int>::max());
                sb->setValue(param.defaultValue.toInt());
                widget = sb;
            } else if (typeId == QMetaType::UInt) {
                auto *sb = new QSpinBox();
                sb->setRange(param.minimum.isValid() ? static_cast<int>(param.minimum.toUInt()) : 0,
                             param.maximum.isValid() ? static_cast<int>(param.maximum.toUInt()) : std::numeric_limits<int>::max());
                sb->setValue(static_cast<int>(param.defaultValue.toUInt()));
                widget = sb;
            } else if (typeId == QMetaType::Double) {
                auto *ssb = new ScientificSpinBox();
                if (param.minimum.isValid()) ssb->setMinimum(param.minimum.toDouble());
                if (param.maximum.isValid()) ssb->setMaximum(param.maximum.toDouble());
                ssb->setValue(param.defaultValue.toDouble());
                widget = ssb;
            } else if (typeId == QMetaType::Bool) {
                auto *cb = new QCheckBox();
                cb->setChecked(param.defaultValue.toBool());
                widget = cb;
            } else {
                auto *le = new QLineEdit();
                le->setText(param.defaultValue.toString());
                widget = le;
            }
            if (widget) {
                d_paramWidgets[param.key] = widget;
                p_requiredParamsLayout->addRow(param.label + ":", widget);
            }
        }

        p_requiredParamsGroup->setTitle("Configuration Parameters");
        p_requiredParamsGroup->setVisible(!params.isEmpty());
        p_importantParamsGroup->setVisible(false);
        p_advancedParamsGroup->setVisible(false);
    }

    adjustSize();
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

    // Write selected protocol and config params to settings before hardware
    // object is created, so the constructor finds the correct values.
    {
        auto selectedProtocol = static_cast<CommunicationProtocol::CommType>(
            p_protocolCombo->currentData().toInt());
        QString settingsKey = BC::Key::hwKey(d_hardwareType, label);
        QSettings s(QCoreApplication::organizationName(), QCoreApplication::applicationName());
        s.beginGroup(settingsKey);
        s.setValue(BC::Key::HW::commType, static_cast<int>(selectedProtocol));

        for (auto it = d_paramWidgets.cbegin(); it != d_paramWidgets.cend(); ++it) {
            QVariant val;
            if (auto *sb = qobject_cast<QSpinBox*>(it.value()))
                val = sb->value();
            else if (auto *ssb = qobject_cast<ScientificSpinBox*>(it.value()))
                val = ssb->value();
            else if (auto *dsb = qobject_cast<QDoubleSpinBox*>(it.value()))
                val = dsb->value();
            else if (auto *cb = qobject_cast<QCheckBox*>(it.value()))
                val = cb->isChecked();
            else if (auto *le = qobject_cast<QLineEdit*>(it.value()))
                val = le->text();

            if (val.isValid())
                s.setValue(it.key(), val);
        }

        s.endGroup();
        s.sync();
    }

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
    QString label = p_labelEdit->text().trimmed();

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
