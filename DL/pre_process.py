# register list
register_Total_list = [ 'rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
                        'eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
                        'ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp', 'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
                        'al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']

# jump instruction list
__jump_ins_list = ['ja', 'jae', 'jb', 'jbe', 'jc', 'jcxz', 'jecxz', 'jrcxz', 'je', 'jg', 'jge', 'jl',
                        'jle', 'jna', 'jnae', 'jnb', 'jnbe', 'jnc', 'jne', 'jng', 'jnge', 'jnl', 'jnle', 'jno',
                        'jnp', 'jns', 'jnz', 'jo', 'jp', 'jpe', 'jpo', 'js', 'jz', 'jmp', 'jmpfi', 'jmpni', 'jmpshort']


def bracket_splict(bracket):#Handling operands enclosed in parentheses
    temp=str(bracket).strip('[]')
    oper=[]
    for i in range(len(temp)):
        if temp[i] == '+' or temp[i] == '-' or temp[i] == '*':
            oper.append(temp[i])

    string_test = temp

    string_List_Split = string_test.split('+')

    string_List_Split = [string_item.split('-') for string_item
                         in string_List_Split]

    string_List_Split_Extend = []

    list(map(string_List_Split_Extend.extend, string_List_Split))

    string_List_Split1 = [string_item.split('*') for string_item
                          in string_List_Split_Extend]

    string_List_Split_Extend1 = []

    list(map(string_List_Split_Extend1.extend, string_List_Split1))

    temp=string_List_Split_Extend1

    return temp,oper

def normalization(i):

    print(i)
    i=i.strip()
    print("origin:",i)
    # temp=i.split(" ")
    # print(temp[0],temp[len(temp)-1])
    i=i.strip()
    instruction=''
    temp = i.split(" ",1)
    oprand=[]
    instruction=instruction+temp[0]+"~"
    #print(temp[0])
    if len(temp) >1:
        oprand=temp[1].strip().split(',')
        #print("oprand: ",oprand)
    else:
        return instruction.strip('~')
    for op in oprand:
        #print(op)
        op=op.strip()

        if i.startswith("call"):## call instruction

            if op.startswith("dword ptr") or op.startswith("word ptr") or op.startswith("byte ptr"):
                strr=''
                temp1=op
                index1=temp1.find('[')
                index2=temp1.find(']')

                brack=temp1[index1 + 1:index2]#[eas+10h----]
                brack,symbol=bracket_splict(brack)

                opst=[]
                cout=0
                for i in brack:
                    if i in register_Total_list:
                        opst.append(i)
                    elif symbol[cout - 1] == '*' and i.isdigit():
                        opst.append('const')
                    elif i.endswith('const') or i.isdigit():
                        opst.append(0)
                    else:
                        opst.append(i)

                for j in range(len(symbol)):
                    strr=strr+str(opst[j])+str(symbol[j])
                instruction= instruction+"ptr["+strr+str(opst[len(opst)-1])+']'

            elif temp[1] in register_Total_list:
                instruction=instruction+temp[1]
            else:
                instruction=instruction+"foo"

        elif op == "0ffffffffh": #Special constant
            instruction = instruction + "0ffffffffh"+","

        elif op.startswith('[') and op.endswith(']'):#ptr
            index1 = op.find('[')
            index2 = op.find(']')

            brack = op[index1 + 1:index2]
            brack, oper = bracket_splict(brack)
            opst = []
            cout=0
            opand1 = ''
            if len(oper) !=0:
                for b in brack:
                    if b in register_Total_list:
                        opst.append(b)
                    elif oper[cout-1] == '*' and b.isdigit():
                        opst.append('const')
                    elif b.isdigit() or b.endswith('h'):
                        opst.append('const')
                    elif b.startswith("var"):
                        opst.append("ptr")
                    elif b.startswith('arg'):
                        opst.append("ptr")
                    else:
                        opst.append("ptr")
                    cout=cout+1
            elif brack[0].startswith('byte@ptr') or brack[0].startswith('word@ptr') or brack[0].startswith('dword@ptr'):
                if brack[0].startswith('byte@ptr'):
                    opand1=opand1+"byte@ptr"
                elif brack[0].startswith('word@ptr'):
                    opand1 = opand1 + "word@ptr"
                elif brack[0].startswith('dword@ptr'):
                    opand1 = opand1 + "dword@ptr"
            else:
                    opand1 = opand1 + brack[0].strip()
            for j in range(len(oper)):
                opand1 = opand1 + str(opst[j]) + str(oper[j])
            if len(oper)!=0:
                opand1 ='['+ opand1+str(opst[len(opst) - 1]) + ']'
            else:
                opand1='['+ opand1+ ']'
            instruction=instruction+opand1+','
            #print(instruction)
        elif op.isdigit() or op.endswith('h'):#no special constant
            if op.startswith('-'):
                instruction=instruction+'-const'+','
            else:
                instruction = instruction + 'const' + ','
            #print(instruction)
        elif op in register_Total_list:#register
            instruction=instruction+op+','
            #print(instruction)
        elif op.startswith("dword@ptr") or op.startswith('byte@ptr') or op.startswith("word@ptr"):#ptr
            strr = ''
            temp1 = op
            index1 = temp1.find('[')
            index2 = temp1.find(']')

            if temp1.find('[') == True:
                brack = temp1[index1 + 1:index2]  # [eas+10h----]
                brack, symbol = bracket_splict(brack)
                # print(brack)
                # print(symbol)
                opst = []
                cout=0
                for i in brack:
                    #print(i)
                    if i in register_Total_list:
                        opst.append(i)
                    elif symbol[cout - 1] == '*' and i.isdigit():
                        opst.append('const')
                    elif i.endswith('h') or i.isdigit():
                        opst.append(0)
                    else:
                        opst.append(i)
                    cout=cout+1
                for j in range(len(symbol)):
                    strr = strr + str(opst[j]) + str(symbol[j])
                if op.startswith("dword"):
                    instruction = instruction + "dword@ptr[" + strr + str(opst[len(opst) - 1]) + ']'+','
                elif op.startswith("word"):
                    instruction = instruction + "word@ptr[" + strr + str(opst[len(opst) - 1]) + ']'+','
                elif op.startswith("byte"):
                    instruction = instruction + "byte@ptr[" + strr + str(opst[len(opst) - 1]) + ']'+','

                #print(instruction)
            else:
                if op.startswith("dword"):
                    instruction = instruction + "dword@ptr" + ','
                elif op.startswith("word"):
                    instruction = instruction + "word@ptr" + ','
                else:
                    instruction = instruction + "byte@ptr" + ','
        elif op.startswith("0x"):
            instruction=instruction+'ptr,'
            #print(instruction)
        elif op.startswith("-0x"):
            instruction=instruction+'-ptr,'
            #print(instruction)
        elif "offset" in op:
            instruction=instruction+'offset,'
            #print(instruction)
        elif i in __jump_ins_list: #jump instruction
            ##short loc_234442
            instruction = instruction + 'ptr,'
            #print(instruction)
        else:
            instruction = instruction + 'tag,' #other case
    return instruction.strip(',')
