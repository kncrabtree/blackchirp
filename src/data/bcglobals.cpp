#include "bcglobals.h"

#include <QStringList>

std::pair<QString, int> BC::Key::parseKey(const QString key)
{
   QStringList l = key.split(hwIndexSep);
   if(l.size() < 2)
       return {key,-1};
   else {
       bool ok = false;
       auto idx = l.at(1).toInt(&ok);
       return {l.at(0),ok ? idx : -1};
   }
}

QString BC::Key::hwKey(const QString k, const int index)
{
    return QString("%1%2%3").arg(k,hwIndexSep,QString::number(index));
}

QString BC::Key::widgetKey(const QString widgetKey, const QString hwKey, const QString hwSubKey)
{
    return QString("%1%2%3%4%5").arg(widgetKey,hwIndexSep,hwKey,hwIndexSep,hwSubKey);
}
