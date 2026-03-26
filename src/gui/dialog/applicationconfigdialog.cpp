#include "applicationconfigdialog.h"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QFont>
#include <QFontDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include <data/storage/applicationconfigmanager.h>
#include <gui/style/themecolors.h>
#include <gui/widget/bcsavepathwidget.h>

ApplicationConfigDialog::ApplicationConfigDialog(bool firstRun, QWidget *parent)
    : QDialog(parent), d_firstRun(firstRun)
{
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg",
                                                ThemeColors::IconPrimary, this));

    if(d_firstRun)
        setWindowTitle("Welcome to Blackchirp - Initial Configuration");
    else
        setWindowTitle("Application Settings");

    auto *mainLayout = new QVBoxLayout(this);

    // --- Application Settings group box ---
    auto *appSettingsGroup = new QGroupBox("Application Settings", this);
    auto *appSettingsLayout = new QVBoxLayout(appSettingsGroup);

    auto& acm = ApplicationConfigManager::instance();
    const auto& options = acm.getOptions();

    for(const auto& option : options)
    {
        int typeId = option.defaultValue.userType();

        if(typeId == QMetaType::Bool)
        {
            QString checkText = option.label;
            if(option.requiresRestart)
                checkText += " (requires restart)";

            auto *cb = new QCheckBox(checkText, this);
            cb->setToolTip(option.description);
            cb->setChecked(acm.getOptionValue(option.settingsKey).toBool());

            const QString key = option.settingsKey;
            connect(cb, &QCheckBox::toggled, this, [this, key](bool checked) {
                d_pendingChanges[key] = checked;
            });

            appSettingsLayout->addWidget(cb);
        }
        else if(typeId == QMetaType::QFont)
        {
            QString labelText = option.label;
            if(option.requiresRestart)
                labelText += " (requires restart)";

            auto *rowLayout = new QHBoxLayout;

            auto *nameLabel = new QLabel(labelText, this);
            rowLayout->addWidget(nameLabel);

            QFont currentFont = acm.getOptionValue(option.settingsKey).value<QFont>();
            QString previewText = QString("%1 %2pt").arg(currentFont.family()).arg(currentFont.pointSize());

            auto *previewLabel = new QLabel(previewText, this);
            previewLabel->setFont(currentFont);
            rowLayout->addWidget(previewLabel, 1);

            auto *changeBtn = new QPushButton("Change...", this);
            rowLayout->addWidget(changeBtn);

            const QString key = option.settingsKey;
            connect(changeBtn, &QPushButton::clicked, this, [this, key, previewLabel]() {
                QFont currentFont = d_pendingChanges.contains(key)
                    ? d_pendingChanges[key].value<QFont>()
                    : ApplicationConfigManager::instance().getOptionValue(key).value<QFont>();

                bool ok = false;
                QFont selectedFont = QFontDialog::getFont(&ok, currentFont, this, "Select Application Font");
                if(ok)
                {
                    d_pendingChanges[key] = selectedFont;
                    previewLabel->setFont(selectedFont);
                    previewLabel->setText(QString("%1 %2pt").arg(selectedFont.family()).arg(selectedFont.pointSize()));
                }
            });

            appSettingsLayout->addLayout(rowLayout);
        }
    }

    mainLayout->addWidget(appSettingsGroup);

    // --- Data Storage group box ---
    auto *storageGroup = new QGroupBox("Data Storage", this);
    auto *storageLayout = new QVBoxLayout(storageGroup);

    p_savePathWidget = new BCSavePathWidget(this);
    storageLayout->addWidget(p_savePathWidget);

    mainLayout->addWidget(storageGroup);

    // --- Button box ---
    p_buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    mainLayout->addWidget(p_buttons);

    connect(p_buttons, &QDialogButtonBox::accepted, this, &ApplicationConfigDialog::accept);
    connect(p_buttons, &QDialogButtonBox::rejected, this, &ApplicationConfigDialog::reject);

    if(d_firstRun)
    {
        p_buttons->button(QDialogButtonBox::Ok)->setEnabled(false);
        connect(p_savePathWidget, &BCSavePathWidget::applied,
                this, [this]() {
            p_buttons->button(QDialogButtonBox::Ok)->setEnabled(true);
        });
    }
}

void ApplicationConfigDialog::accept()
{
    auto& acm = ApplicationConfigManager::instance();
    bool needsRestart = false;

    for(auto it = d_pendingChanges.constBegin(); it != d_pendingChanges.constEnd(); ++it)
    {
        acm.setOptionValue(it.key(), it.value());

        for(const auto& opt : acm.getOptions())
        {
            if(opt.settingsKey == it.key() && opt.requiresRestart)
            {
                needsRestart = true;
                break;
            }
        }
    }

    if(p_savePathWidget->isReady())
        p_savePathWidget->save();

    if(needsRestart && !d_firstRun)
    {
        QMessageBox::information(this, "Restart Required",
            "Some settings require a restart to take effect.");
    }

    QDialog::accept();
}
