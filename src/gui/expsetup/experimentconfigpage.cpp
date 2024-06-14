#include "experimentconfigpage.h"

ExperimentConfigPage::ExperimentConfigPage(const QString key, const QString title, Experiment *exp, QWidget *parent)
    : QWidget{parent}, SettingsStorage{key}, d_key{key}, d_title{title}, p_exp{exp}
{

}
