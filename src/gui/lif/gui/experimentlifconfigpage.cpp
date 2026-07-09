#include "experimentlifconfigpage.h"

#include <QAbstractButton>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

#include <data/experiment/hardwaredatacontainer.h>
#include <data/loadout/loadoutmanager.h>
#include <gui/lif/gui/lifconfigwidget.h>
#include <gui/lif/gui/lifcontrolwidget.h>
#include <gui/lif/gui/lifconversionwidget.h>

using namespace BC::Key::WizLif;
using namespace Qt::StringLiterals;

ExperimentLifConfigPage::ExperimentLifConfigPage(Experiment *exp, QWidget *parent) :
    ExperimentConfigPage(key,title,exp,parent)
{
    // Look up LIF scope and laser hardware keys from experiment's hardware data
    QString digitizerHwKey;
    QString laserHwKey;
    for (auto it = exp->d_hardwareData.hardwareMap.cbegin();
         it != exp->d_hardwareData.hardwareMap.cend(); ++it) {
        if (it.value().type == BC::Data::HardwareType::LifDigitizer)
            digitizerHwKey = it.key();
        else if (it.value().type == BC::Data::HardwareType::LifLaser)
            laserHwKey = it.key();
    }

    p_widget = new LifConfigWidget(digitizerHwKey, laserHwKey, false, this);

    auto vbl = new QVBoxLayout;
    vbl->addWidget(p_widget);

    setLayout(vbl);

    // For an existing experiment, the conversion table shows the topology
    // actually recorded for it (loaded from liftopology.csv on disk, or
    // seeded from the current LIF preset when Experiment::enableLif() ran
    // for a repeat/new experiment earlier in the wizard flow). A genuinely
    // new experiment has no LifConfig yet at page-construction time (this
    // page is built before ExperimentTypePage::apply() first runs
    // enableLif()); LifConversionWidget's own constructor already seeds
    // itself from the current LIF preset in that case, exactly mirroring
    // FtmwConfigWidget/ExperimentFtmwConfigPage.
    if(p_exp->d_number > 0 && p_exp->lifEnabled())
        p_widget->setFromConfig(*p_exp->lifConfig());

    connect(p_widget, &LifConfigWidget::edited,
            this, &ExperimentLifConfigPage::presetChanged);
}

LifControlWidget *ExperimentLifConfigPage::lifControlWidget()
{
    return p_widget->lifControlWidget();
}

LifConversionWidget *ExperimentLifConfigPage::lifConversionWidget()
{
    return p_widget->lifConversionWidget();
}

void ExperimentLifConfigPage::initialize()
{
}

bool ExperimentLifConfigPage::validate()
{
    if(!p_exp->lifEnabled())
        return true;

    auto result = p_widget->lifConversionWidget()->model()->assemblyResult();
    if(!result.ok)
    {
        emit error(result.errorString);
        return false;
    }

    return true;
}

void ExperimentLifConfigPage::apply()
{
    if(isEnabled() && p_exp->lifEnabled())
        p_widget->toConfig(*p_exp->lifConfig());
}

void ExperimentLifConfigPage::commitLifPreset()
{
    if(!isEnabled() || !p_widget->isDirty())
        return;

    const auto activeName = LoadoutManager::instance().currentLoadoutName();
    if(activeName.isEmpty())
        return;

    const auto currentPresetName = LoadoutManager::instance().currentLifPresetName(activeName);
    const bool canOverwrite = !currentPresetName.isEmpty()
        && currentPresetName != BC::Store::LM::lastUsedLifPresetName;

    QMessageBox msgBox(this);
    msgBox.setWindowTitle(u"Save LIF changes?"_s);
    msgBox.setText(u"The LIF configuration has unsaved changes."_s);

    const QString overwriteLabel = canOverwrite
        ? QString(u"Overwrite \"%1\""_s).arg(currentPresetName)
        : u"Overwrite current preset"_s;
    QAbstractButton *overwriteBtn = static_cast<QAbstractButton*>(
        msgBox.addButton(overwriteLabel, QMessageBox::AcceptRole));
    overwriteBtn->setEnabled(canOverwrite);
    QAbstractButton *saveAsBtn = static_cast<QAbstractButton*>(
        msgBox.addButton(u"Save as new preset..."_s, QMessageBox::ActionRole));
    msgBox.addButton(u"Proceed without saving"_s, QMessageBox::DestructiveRole);

    msgBox.exec();
    auto *clicked = msgBox.clickedButton();
    const auto preset = p_widget->toLifPreset();

    if(clicked == overwriteBtn)
    {
        LoadoutManager::instance().putLifPreset(activeName, currentPresetName, preset);
        LoadoutManager::instance().putLifPreset(
            activeName, BC::Store::LM::lastUsedLifPresetName, preset);
        p_widget->clearDirty();
    }
    else if(clicked == saveAsBtn)
    {
        bool ok;
        auto name = QInputDialog::getText(
            this, u"Save LIF Preset As"_s, u"Preset name:"_s,
            QLineEdit::Normal, {}, &ok).trimmed();

        bool saved = false;
        if(ok && !name.isEmpty() && name != BC::Store::LM::lastUsedLifPresetName)
        {
            bool doSave = true;
            if(LoadoutManager::instance().lifPresetExists(activeName, name))
            {
                const auto r = QMessageBox::question(
                    this, u"Overwrite Preset"_s,
                    QString(u"Preset \"%1\" already exists. Overwrite?"_s).arg(name),
                    QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
                doSave = (r == QMessageBox::Yes);
            }
            if(doSave)
            {
                LoadoutManager::instance().putLifPreset(activeName, name, preset);
                LoadoutManager::instance().putLifPreset(
                    activeName, BC::Store::LM::lastUsedLifPresetName, preset);
                LoadoutManager::instance().setCurrentLifPresetName(activeName, name);
                saved = true;
            }
        }

        if(!saved)
        {
            // Sub-dialog cancelled, invalid name, or overwrite declined — proceed without saving
            LoadoutManager::instance().putLifPreset(
                activeName, BC::Store::LM::lastUsedLifPresetName, preset);
            LoadoutManager::instance().setCurrentLifPresetName(
                activeName, BC::Store::LM::lastUsedLifPresetName);
        }
        p_widget->clearDirty();
    }
    else
    {
        // Proceed without saving
        LoadoutManager::instance().putLifPreset(
            activeName, BC::Store::LM::lastUsedLifPresetName, preset);
        LoadoutManager::instance().setCurrentLifPresetName(
            activeName, BC::Store::LM::lastUsedLifPresetName);
        p_widget->clearDirty();
    }
}
