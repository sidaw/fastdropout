function acc = evalconll(iid, ypredtst, ytesttrue, resultname, paramstring, outputname, isdev)

acc = sum(ypredtst==oneofktoscalar(ytesttrue)) / size(ytesttrue,1);

if iid == 0
    fname = [resultname '.tst'];
    save(fname, 'ypredtst', '-ascii');
    lblcmd = ['echo ' paramstring '>> ' outputname];
    if isdev
        fieldname = 'devfields';
    else
        fieldname = 'testfields';
    end
    pycommandtst = ...
        ['./data/conll-ner/generateconnloutput.py '...
        fname ' data/conll-ner/' fieldname ' >' resultname '.conlltest'];
    perlcommandtst = ['./data/conll-ner/conlleval.pl <' resultname '.conlltest' '>> ' outputname];
    
    unix([lblcmd ';' pycommandtst ';' perlcommandtst]);
    
else
    tokenmap = {'B-LOC', 'B-MISC', 'B-ORG', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O'};
    ytesttrue = oneofktoscalar(ytest);
    fname = [resultname '.tst.idd'];
    file = fopen(fname, 'w+');
    for j = 1:length(ytesttrue)
        fprintf(file, 'token%d %s %s\n', j, tokenmap{ytesttrue(j)}, tokenmap{ypredtst(j)} );
    end
    perlcommandtst = ['./data/conll-ner/conlleval.pl -r -o 8 < ' fname '>> iid' outputname];
    lblcmd = ['echo idd' paramstring '>> iid' outputname];
    unix([lblcmd ';' perlcommandtst]);
   
end

end