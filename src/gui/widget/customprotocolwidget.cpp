#include <gui/widget/customprotocolwidget.h>
#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>
#include <hardware/core/hardwareregistry.h>

#include <QVBoxLayout>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QSettings>
#include <QCoreApplication>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static QVector<CustomCommDef> loadDefs(const QString& hwType, const QString& hwImpl)
{
    return HardwareRegistry::instance().getCustomCommDefs(hwType, hwImpl);
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

CustomProtocolWidget::CustomProtocolWidget(const QString& hardwareKey, QWidget *parent)
    : ProtocolWidget(hardwareKey, parent)
{
    setupUI();

    // Derive hwType and implementation from the per-profile storage key.
    auto [hwType, label] = BC::Key::parseKey(hardwareKey);
    Q_UNUSED(label)
    SettingsStorage hwSettings(hardwareKey, SettingsStorage::Hardware);
    QString hwImpl = hwSettings.get(BC::Key::HW::model, QString());

    generateDynamicUI(loadDefs(hwType, hwImpl));
    loadProtocolSettings();
}

CustomProtocolWidget::CustomProtocolWidget(const QString& hwType, const QString& hwImpl,
                                           QWidget *parent)
    : ProtocolWidget(QString(), parent)
{
    setupUI();
    generateDynamicUI(loadDefs(hwType, hwImpl));
    // No values to load — fields remain at their type-appropriate defaults.
}

// ---------------------------------------------------------------------------
// UI setup
// ---------------------------------------------------------------------------

void CustomProtocolWidget::setupUI()
{
    p_layout = new QVBoxLayout(this);

    auto *headerLabel = new QLabel("<b>Custom Protocol Configuration</b>", this);
    p_layout->addWidget(headerLabel);

    p_formLayout = new QFormLayout();
    p_layout->addLayout(p_formLayout);

    p_layout->addStretch();
}

void CustomProtocolWidget::generateDynamicUI(const QVector<CustomCommDef>& defs)
{
    clearDynamicUI();

    if (defs.isEmpty()) {
        auto *noSettingsLabel = new QLabel("No custom settings defined for this hardware.", this);
        noSettingsLabel->setStyleSheet("color: gray; font-style: italic;");
        p_formLayout->addRow(noSettingsLabel);
        d_dynamicWidgets.append(noSettingsLabel);
        return;
    }

    for (const auto& def : defs) {
        switch (def.type) {
        case CustomCommType::String: {
            auto *lineEdit = new QLineEdit(this);
            int maxLen = def.bound.isValid() ? def.bound.toInt() : 100;
            lineEdit->setMaxLength(maxLen);
            if (!def.description.isEmpty())
                lineEdit->setToolTip(def.description);
            p_formLayout->addRow(def.label + ":", lineEdit);
            d_stringFields.append(qMakePair(def.key, lineEdit));
            d_dynamicWidgets.append(lineEdit);
            break;
        }
        case CustomCommType::Int: {
            auto *spinBox = new QSpinBox(this);
            int minVal = def.bound.isValid()  ? def.bound.toInt()  : 0;
            int maxVal = def.bound2.isValid() ? def.bound2.toInt() : 100000;
            spinBox->setRange(minVal, maxVal);
            spinBox->setValue(minVal);
            if (!def.description.isEmpty())
                spinBox->setToolTip(def.description);
            p_formLayout->addRow(def.label + ":", spinBox);
            d_intFields.append(qMakePair(def.key, spinBox));
            d_dynamicWidgets.append(spinBox);
            break;
        }
        case CustomCommType::FilePath: {
            auto *rowWidget = new QWidget(this);
            auto *rowLayout = new QHBoxLayout(rowWidget);
            rowLayout->setContentsMargins(0, 0, 0, 0);
            auto *lineEdit = new QLineEdit(rowWidget);
            if (!def.description.isEmpty())
                lineEdit->setToolTip(def.description);
            auto *browseBtn = new QPushButton("Browse…", rowWidget);
            rowLayout->addWidget(lineEdit);
            rowLayout->addWidget(browseBtn);
            connect(browseBtn, &QPushButton::clicked, this, [lineEdit, this]() {
                QString path = QFileDialog::getOpenFileName(this, tr("Select File"), lineEdit->text());
                if (!path.isEmpty())
                    lineEdit->setText(path);
            });
            p_formLayout->addRow(def.label + ":", rowWidget);
            d_filePathFields.append(qMakePair(def.key, lineEdit));
            d_dynamicWidgets.append(rowWidget);
            break;
        }
        }
    }
}

void CustomProtocolWidget::clearDynamicUI()
{
    for (auto *widget : d_dynamicWidgets) {
        p_formLayout->removeWidget(widget);
        widget->deleteLater();
    }
    d_dynamicWidgets.clear();
    d_stringFields.clear();
    d_intFields.clear();
    d_filePathFields.clear();
}

// ---------------------------------------------------------------------------
// Load / save
// ---------------------------------------------------------------------------

void CustomProtocolWidget::loadProtocolSettings()
{
    for (const auto& [key, lineEdit] : d_stringFields)
        lineEdit->setText(getGroupValue<QString>(BC::Key::Comm::custom, key, QString()));

    for (const auto& [key, spinBox] : d_intFields)
        spinBox->setValue(getGroupValue<int>(BC::Key::Comm::custom, key, spinBox->minimum()));

    for (const auto& [key, lineEdit] : d_filePathFields)
        lineEdit->setText(getGroupValue<QString>(BC::Key::Comm::custom, key, QString()));
}

void CustomProtocolWidget::saveProtocolSpecificSettings()
{
    for (const auto& [key, lineEdit] : d_stringFields)
        setGroupValue(BC::Key::Comm::custom, key, lineEdit->text());

    for (const auto& [key, spinBox] : d_intFields)
        setGroupValue(BC::Key::Comm::custom, key, spinBox->value());

    for (const auto& [key, lineEdit] : d_filePathFields)
        setGroupValue(BC::Key::Comm::custom, key, lineEdit->text());
}

void CustomProtocolWidget::saveToStorage(const QString& storageKey) const
{
    QSettings s(QCoreApplication::organizationName(), QCoreApplication::applicationName());
    s.beginGroup(storageKey);
    s.beginGroup(BC::Key::Comm::custom);

    for (const auto& [key, lineEdit] : d_stringFields)
        s.setValue(key, lineEdit->text());

    for (const auto& [key, spinBox] : d_intFields)
        s.setValue(key, spinBox->value());

    for (const auto& [key, lineEdit] : d_filePathFields)
        s.setValue(key, lineEdit->text());

    s.endGroup();
    s.endGroup();
    s.sync();
}
