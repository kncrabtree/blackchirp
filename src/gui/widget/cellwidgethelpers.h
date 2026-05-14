#ifndef CELLWIDGETHELPERS_H
#define CELLWIDGETHELPERS_H

class QTableWidget;
class QWidget;

namespace BC::Gui {

void centerCellWidget(QTableWidget *table, int row, int col, QWidget *w);

} // namespace BC::Gui

#endif // CELLWIDGETHELPERS_H
