#include "experimentconfigpage.h"

ExperimentConfigPage::ExperimentConfigPage(QString key, QString title, Experiment *exp, QWidget *parent)
    : QWidget{parent}, SettingsStorage{key}, d_title{title}, p_exp{exp}
{

}
